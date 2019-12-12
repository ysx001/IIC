function [result] = feature_process_3d(features, nx2, ny2, nf2)
% f - nx - ny
% f = permute(features(:,:),[2 1]);


% features: nd - nx - ny - nz
[nf, nx, ny, nz] = size(features);
result = zeros(nx2, ny2, nz, nf2);
%% PCA compresss features
for i = 1 : nz
    f = features(:,:,:,i); % nd - nx - ny
    f = permute(f(:,:),[2 1]); % nx*ny - nd
    coeff = pca(f); 
    f = reshape(f * coeff(:,1:nf2), [nx, ny, nf2]);
    result(:,:,i,:) = imresize(f, [nx2, ny2]);
end


%% normalize the feature
for i = 1 : nf2
    for j = 1 : nz
        s = max(max(abs(result(:,:,j,i)),[],2),[],1) + eps;
        result(:,:,j,i) = result(:,:,j,i)/ s;
    end
end

end
