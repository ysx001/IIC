function res = zero_pad(im,b)
[nx, ny, nz, nc] = size(im);
res = zeros(nx + 2 * b , ny + 2 * b, nz, nc);
res((b+1):(nx+b),(b+1):(ny+b),:,:) = im;
end

