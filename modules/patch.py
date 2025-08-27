class PatchExpand(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expansion = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = self.norm(x)
        x = self.expansion(x)

        x = x.reshape(B, D, H, W, 2, 2, C//2)
        x = rearrange(x, 'b d h w h1 w1 c -> b d h h1 w w1 c')
        x = x.reshape(B, D, 2*H, 2*W, C//2)

        return x
    

class AllPatchExpand(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand1 = PatchExpand(dim, norm_layer)
        self.expand2 = PatchExpand(dim, norm_layer)
        self.expand3 = PatchExpand(dim, norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        x1 = rearrange(x[0], 'b c d h w -> b d h w c')
        x2 = rearrange(x[1], 'b c d h w -> b d h w c')
        gate = rearrange(x[2], 'b c d h w -> b d h w c')
        x1 = self.expand1(x1)
        x2 = self.expand2(x2)
        gate = torch.sigmoid(self.expand3(gate))
        x1 = rearrange(x1, 'b d h w c -> b c d h w')
        x2 = rearrange(x2, 'b d h w c -> b c d h w')
        gate = rearrange(gate, 'b d h w c -> b c d h w')
        return [x1, x2, gate]
    

class PatchEmbed(nn.Module):

    def __init__(self, img_size=(69,721,1440), embed_dim=96, patch_size=(1,4,4), norm_layer=None):
        super().__init__()

        if img_size[1] % 2:
            self.proj3d = nn.Conv3d(5, embed_dim, kernel_size=(patch_size[0], patch_size[1]+1, patch_size[2]), stride=patch_size)
            self.proj2d = nn.Conv2d(4, embed_dim, kernel_size=(patch_size[1]+1, patch_size[2]), stride=(patch_size[1], patch_size[2]))
        else:
            self.proj3d = nn.Conv3d(5, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.proj2d = nn.Conv2d(4, embed_dim, kernel_size=patch_size[1:], stride=patch_size[1:])

        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        B, C, H, W = x.shape
        # print(x.shape)#[1,69,121,240]\
        # exit()
        x2d = x[:,:4,:,:]  # b,4,721,1440
        x3d = x[:,4:,:,:].reshape(B, 5, C//5, H, W)  # b,5,13,721,1440
        x2d = self.proj2d(x2d).unsqueeze(2)  # b,c,1,180,360
        x3d = self.proj3d(x3d)  # b,c,13,180,360
        x = torch.cat([x3d, x2d], dim=2)  # b,c,14,180,360

        if self.norm is not None:
            D, H, W = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        return x


class AllPatchEmbed(nn.Module):

    def __init__(self, img_size=(69,721,1440), embed_dim=96, patch_size=(1,4,4), norm_layer=None):
        super().__init__()

        self.patch1 = PatchEmbed(img_size=img_size, embed_dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)
        self.patch2 = PatchEmbed(img_size=img_size, embed_dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)
        self.patch3 = PatchEmbed(img_size=img_size, embed_dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)

        self.patch_resolution = (img_size[0]//5+1, img_size[1]//patch_size[1], img_size[2]//patch_size[2])

    def forward(self, x1, x2, mask):
        """Forward function."""

        x1 = self.patch1(x1)
        x2 = self.patch2(x2)
        mask = torch.sigmoid(self.patch3(mask))
        
        return [x1, x2, mask]


class PatchRecover(nn.Module):

    def __init__(self, img_size=(69,721,1440), embed_dim=96, patch_size=(1,4,4)):
        super().__init__()

        if img_size[1] % 2:
            self.proj3d = nn.ConvTranspose3d(embed_dim, 5, kernel_size=(patch_size[0], patch_size[1]+1, patch_size[2]), stride=patch_size)
            self.proj2d = nn.ConvTranspose2d(embed_dim, 4, kernel_size=(patch_size[1]+1, patch_size[2]), stride=(patch_size[1], patch_size[2]))
        else:
            self.proj3d = nn.ConvTranspose3d(embed_dim, 5, kernel_size=patch_size, stride=patch_size)
            self.proj2d = nn.ConvTranspose2d(embed_dim, 4, kernel_size=patch_size[1:], stride=patch_size[1:])

    def forward(self, x):
        """Forward function."""

        x2d = x[:,:,-1:,:,:].squeeze(2)  # b,c,180,360
        x3d = x[:,:,:-1,:,:]  # b,c,13,180,360
        x2d = self.proj2d(x2d)  # b,4,721,1440
        x3d = self.proj3d(x3d).flatten(1,2)  # b,65,721,1440
        x = torch.cat([x2d, x3d], dim=1)

        return x