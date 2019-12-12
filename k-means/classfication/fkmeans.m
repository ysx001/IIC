load('/Users/huyx11/Downloads/data.mat')
slice = 40;
[nx, ny, nz, nc] = size(imgs);
x = imgs(:,:,slice,:);
x = permute(x, [4 1 2 3]);
x = permute(x(:,:),[2 1]);
res5 = reshape(kmeans(x(:,1), 5),[nx, ny]);
figure,
imshow(abs([res5']),[]),colormap jet, title('K-means, 5 classes')



%%
figure,subplot(221),
imshow(log(abs(log(segs(:,:,slice,2))))',[]), colormap jet, title('FreeSurfer')
subplot(222),
imshow(abs([res2']),[]),colormap jet, title('K-means, 4 classes')
subplot(223),
imshow(abs([res3']),[]),colormap jet, title('K-means, 5 classes')
subplot(224),
imshow(abs([res']),[]),colormap jet, title('K-means, 10 classes')


%%
im = imread('/Users/huyx11/Downloads/myelin.png');
im = im(:,:,1);
winsize = [3, 3];
im2 = im2row(im, winsize);
[nx, ny] = size(im);
res5 = reshape(kmeans(im2, 3),[nx - winsize(1) + 1, ny - winsize(2) + 1]);
figure,imshow(res5,[])

%%
figure,subplot(121), imagesc(im),title('myelin')
subplot(122),imagesc(res5), title('k-means, 3 classes')
