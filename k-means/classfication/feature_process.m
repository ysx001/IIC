function [res] = feature_process(features)
% f - nx - ny
% f = permute(features(:,:),[2 1]);
res = pca(features(:,:));
end

