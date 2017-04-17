function CNN_3D(imdb,varargin)
% EXERCISE4   Part 4 of the VGG CNN practical

 %setup;
%  pause
% % -------------------------------------------------------------------------
% Part 4.1: Load the data
% -------------------------------------------------------------------------
% imdb.images.labels=changem(imdb.images.labels,[-1 1],[0 1]);
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = initialize3DCNN() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 500 ;
trainOpts.numEpochs = 6 ;
trainOpts.continue = true ;
trainOpts.useGpu = false;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'data/CNN_3D2' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
% imdb = load('data/phantom6.mat') ;
% imdb=im.phantom;
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;
% 
% Convert to a GPU array if needed
% if trainOpts.useGpu
%   imdb.images.data = gpuArray(imdb.images.data) ;
% end
h=@getBatch;
% Call training function in MatConvNet
[net,info] = cnn_train3D(net, imdb, h, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
% if trainOpts.useGpu
%   net = vl_simplenn_move(net, 'cpu') ;
% end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data/CNN3D/phantom_net.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Part 4.5: apply the model
% -------------------------------------------------------------------------
% 
% % Load the CNN learned before
%  net = load('data/phantom3/phantom_net.mat') ;
% % net = load('data/chars-experiment/charscnn-jit.mat') ;

% % Load the sentence
% i=load('data/phantom2');
% in=i.images.data;
% 
% % im = im2single(imread('data/sentence-lato.png')) ;
% im = single(in) ;
% 
% im = 256*(im - net.imageMean) ;
% for j=1:262144;
% % Apply the CNN to the larger image
% res = vl_simplenn(net, gpuArray(im(:,:,j))) ;
% for i=1:size(res(end).x,2)
%   [score(i),pred(i)] = max(squeeze(res(end).x(1,i,:))) ;
% end
% p=gather(pred);
% predict(j)=p;
% end
% seg=reshape(predict,[512 512]);
% figure(3);imshow(seg,[])

% % Visualize the results
% imdb = load('data/charsdb.mat') ;
% 
% figure(3) ; clf ;
% decodeCharacters(net, imdb, im, res) ;
% % 
% % -------------------------------------------------------------------------
% % Part 4.6: train with jitter
% % -------------------------------------------------------------------------
% 
% trainOpts.batchSize = 100 ;
% trainOpts.numEpochs = 15 ;
% trainOpts.continue = true ;
% trainOpts.learningRate = 0.001 ;
% trainOpts.expDir = 'data/chars-jit-experiment' ;
% 
% % Initlialize a new network
% net = initializeCharacterCNN() ;
% 
% % Call training function in MatConvNet
% [net,info] = cnn_train(net, imdb, @getBatchWithJitter, trainOpts) ;
% 
% % Move the CNN back to CPU if it was trained on GPU
% if trainOpts.useGpu
%   net = vl_simplenn_move(net, 'cpu') ;
% end
% 
% % Save the result for later use
% net.layers(end) = [] ;
% net.imageMean = imageMean ;
% save('data/chars-experiment/charscnn-jit.mat', '-struct', 'net') ;
% 
% % Visualize the results on the sentence
% figure(4) ; clf ;
% decodeCharacters(net, imdb, im, vl_simplenn(net, im)) ;
% 
% --------------------------------------------------------------------
function [im, label] = getBatch(imdb, batch)
% --------------------------------------------------------------------
[a b c d]=size(imdb.images.data);
im = imdb.images.data(:,:,:,batch) ;
im = reshape(im, a, b, c,1, []) ;
label = imdb.images.labels(:,batch) ;



% --------------------------------------------------------------------
% function [im, labels] = getBatchWithJitter(imdb, batch)
% % --------------------------------------------------------------------
% im = imdb.images.data(:,:,batch) ;
% labels = imdb.images.label(1,batch) ;
% 
% n = numel(batch) ;
% train = find(imdb.images.set == 1) ;
% 
% sel = randperm(numel(train), n) ;
% im1 = imdb.images.data(:,:,sel) ;
% 
% sel = randperm(numel(train), n) ;
% im2 = imdb.images.data(:,:,sel) ;
% 
% ctx = [im1 im2] ;
% ctx(:,17:48,:) = min(ctx(:,17:48,:), im) ;
% 
% dx = randi(11) - 6 ;
% im = ctx(:,(17:48)+dx,:) ;
% sx = (17:48) + dx ;
% 
% dy = randi(5) - 2 ;
% sy = max(1, min(32, (1:32) + dy)) ;
% 
% im = ctx(sy,sx,:) ;
% 
% % Visualize the batch:
% % figure(100) ; clf ;
% % vl_imarraysc(im) ;
% 
% im = 256 * reshape(im, 32, 32, 1, []) ;
% 
% 
% vl_simplenn_display(net)
