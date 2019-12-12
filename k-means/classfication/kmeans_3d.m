function [result] = kmeans_3d(imgs, numk, patch)
% numk: number of classes
% patch: patch size (odd)

[nx, ny, nz, nc] = size(imgs);
result = zeros(nx, ny, nz);

imgs = zero_pad(imgs, (patch-1)/2); % nx - ny - nz - nc
imgs = permute(imgs, [1 2 4 3]); % nx - ny - nc - nz
for i = 1 : nz
    x = imgs(:,:,:,i);
    x = im2row(x, [patch, patch]);
    result(:,:,i) = reshape(kmeans(x(:,:), numk),[nx, ny]);
    
end

end
