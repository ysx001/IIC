load('/Users/huyx11/Downloads/combined_data.mat')
slice = 40;
winsize = [3, 3];
[nx, ny, nz, nc] = size(imgs);
x = squeeze(imgs(:,:,slice,:));
x = zero_pad(x, 1);
x = im2row(x, winsize);
%x2 = std(x, 1, 2);
types = [5, 10, 15, 20];
types = [4:10];
for i = 1 : length(types)
    %p(:,:,i) = reshape(kmeans(x(:,:), types(i)),[nx - winsize(1) + 1, ny - winsize(2) + 1]);
        p(:,:,i) = reshape(kmeans(x(:,:), types(i)),[nx, ny]);

end

%res5 = reshape(kmeans([x(:,:) x2(:,:)], 5),[nx - winsize(1) + 1, ny - winsize(2) + 1]);
% figure,
% imshow(abs([res5']),[]),colormap jet, title('K-means, 5 classes')


%%
figure,subplot(221),
imshow(log(abs(log(segs(:,:,slice,2))))',[]), colormap jet, title('FreeSurfer')
subplot(222),
imshow(abs([res4']),[]),colormap jet, title('K-means, 4 classes')
subplot(223),
imshow(abs([res5']),[]),colormap jet, title('K-means, 5 classes')
subplot(224),
imshow(abs([res10']),[]),colormap jet, title('K-means, 10 classes')


%%
load('/Users/huyx11/Downloads/data.mat')
slice = 40;
[nx, ny, nz, nc] = size(imgs);
x = imgs(:,:,slice,:);
x = permute(x, [4 1 2 3]);
x = permute(x(:,:),[2 1]);
types = [5, 10, 15, 20];
for i = 1 : length(types)
    r(:,:,i) = reshape(kmeans(x(:,:), types(i)),[nx, ny]);
end
figure,
imshow(abs([r5']),[]),colormap jet, title('K-means, 5 classes')


%%

figure,subplot(221),
imshow(log(abs(log(segs(:,:,slice,2))))',[]), colormap jet, title('FreeSurfer')
subplot(222),
imshow(abs([r4']),[]),colormap jet, title('K-means, 4 classes')
subplot(223),
imshow(abs([r5']),[]),colormap jet, title('K-means, 5 classes')
subplot(224),
imshow(abs([r10']),[]),colormap jet, title('K-means, 10 classes')


%%
save('result','r4','r5','r10','res4','res5','res10','v4','v5','v10')




%% adding vgg features
load('/Users/huyx11/Downloads/data.mat')
slice = 40;
[nx, ny, nz, nc] = size(imgs);

load('/Users/huyx11/Downloads/features_40.mat')
f = feature_process(features);
f = reshape(f(:,1:5),[56, 56, 5]);
f = imresize(f, [nx, ny]);

x = squeeze(imgs(:,:,slice,:));
x = cat(3, x, f);
x = permute(x, [3 1 2]);
x = permute(x(:,:),[2 1]);


types = [5, 10, 15, 20];
for i = 1 : length(types)
    v(:,:,i) = reshape(kmeans(x(:,:), types(i)),[nx, ny]);
end

figure,
imshow(abs([v10']),[]),colormap jet, title('K-means, 5 classes')

%%
figure,subplot(221),
imshow(log(abs(log(segs(:,:,slice,2))))',[]), colormap jet, title('FreeSurfer')
subplot(222),
imshow(abs([v4']),[]),colormap jet, title('K-means, 4 classes')
subplot(223),
imshow(abs([v5']),[]),colormap jet, title('K-means, 5 classes')
subplot(224),
imshow(abs([v10']),[]),colormap jet, title('K-means, 10 classes')


%% patch + vgg features
load('/Users/huyx11/Downloads/data.mat')
slice = 40;
[nx, ny, nz, nc] = size(imgs);

load('/Users/huyx11/Downloads/features_40.mat')
f = feature_process(features);
f = reshape(f(:,1:5),[56, 56, 5]);
f = imresize(f, [nx, ny]);

x = squeeze(imgs(:,:,slice,:));
x = cat(3, x, f);

x = im2row(x, winsize);
%x2 = std(x, 1, 2);

vp5 = reshape(kmeans([x(:,:)], 5),[nx - winsize(1) + 1, ny - winsize(2) + 1]);
figure,
imshow(abs([vp5']),[]),colormap jet, title('K-means, 5 classes')


%%
figure,subplot(221),
imshow(log(abs(log(segs(:,:,slice,2))))',[]), colormap jet, title('FreeSurfer')
subplot(222),
imshow(abs([vp4']),[]),colormap jet, title('K-means, 4 classes')
subplot(223),
imshow(abs([vp5']),[]),colormap jet, title('K-means, 5 classes')
subplot(224),
imshow(abs([vp10']),[]),colormap jet, title('K-means, 10 classes')

