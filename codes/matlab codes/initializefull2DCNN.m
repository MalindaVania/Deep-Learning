function net = initializefull2DCNN()

f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(11,11,1,300, 'single'), ...
                           'biases', zeros(1, 300, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'convt', ...
                           'filters', f*randn(11,11,3,300, 'single'),...
                            'biases', zeros(1, 3, 'single'), ...
                           'upsample',1);
% net.layers{end+1} = struct('type', 'conv', ...
%                            'filters', f*randn(5,5,20,50, 'single'),...
%                            'biases', zeros(1,50,'single'), ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'filters', f*randn(4,4,50,500, 'single'),...
%                            'biases', zeros(1,500,'single'), ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'filters', f*randn(2,2,500,3, 'single'),...
%                            'biases', zeros(1,3,'single'), ...
%                            'stride', 1, ...
%                            'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;