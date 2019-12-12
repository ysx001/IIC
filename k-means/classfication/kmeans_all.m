% 0: mask, 1-4: grey matter, white matter, deep grey matter, csf

pathname = '/bmrNAS/people/yuxinh/DL_diffseg/DiffSeg-Data';
files = dir([pathname,'/*Case*']);
numk = 5;
numf = 1;
for i = 1 : length(files)
    load([pathname, '/', files(i).name,'/data.mat'])
    load([pathname, '/', files(i).name,'/features.mat'])
    imgs = img;
    f2 = feature_process_3d(features, size(imgs,1), size(imgs,2), numf)/2;
    m1 = kmeans_3d(imgs, numk, 1);
    m2 = kmeans_3d(imgs, numk, 3);
    m3 = kmeans_3d(imgs, numk, 5);
    m4 = kmeans_3d(imgs, numk, 7);
    m5 = kmeans_3d(cat(4, imgs, f2), numk, 1);
    m6 = kmeans_3d(cat(4, imgs, f2), numk, 3);
    m0 = combinedsegs(:,:,:,2);
    save([pathname, '/', files(i).name,'/kmeans'],'m0','m1','m2','m3','m4','m5','m6')

end

% method1: point kmeans
% method2: patch-3 kmeans
% method3: patch-5 kmeans
% method4: patch-7 kmeans
% method5: vgg kmeans
% method6: patch-3 + vgg kmeans



%%
s = 112;
m0 = mask;
figure,imshow([m0(:,:,s)*5 m1(:,:,s) m2(:,:,s) m3(:,:,s);...
    m4(:,:,s) m5(:,:,s) m6(:,:,s) m5(:,:,s);],[]),colormap jet

%%
pathname = '/bmrNAS/people/yuxinh/DL_diffseg/MSSeg-Data';
files = dir([pathname,'/*FLAIR*']);
for i = 1 : length(files)
    %%unzip([pathname,'/', files(i).name(1:end-7)]);
    name = files(i).name(1:end-10);
    mkdir([pathname,'/', name]);
    if exist([pathname, '/', name,'_lesion.mat'], 'file') == 2
        load([pathname, '/', name,'_lesion.mat'])
        mask = img;
        load([pathname, '/', files(i).name])
        save([pathname, '/', name,'/data'],'img','mask')
    else
        load([pathname, '/', files(i).name])
        save([pathname, '/', name,'/data'],'img')
    end
end
