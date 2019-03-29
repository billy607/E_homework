clear all;
%-------------------------------------------------------------------------------
% Input parameters:
number_patches = 1000; % Number of patches chosen from image
patch = 16;             % Size of each square patch
scratch = 4;            % Scratch number of pixels in final display
number_eig = 64;        % Number of eigenvectors in final display
%-------------------------------------------------------------------------------
% Image I/O
x = imread('clockwork-angels.jpg'); % Input image
x = double(x);                      % Conversion to double precision
xr = x(:,:,1);                      % Choosing red channel
[m,n] = size(xr);                   % Size of input image
%-------------------------------------------------------------------------------
% Generating random patch location indices
xi = rand(number_patches,1);        % Random number generation for row index
yi = rand(number_patches,1);        % Random number generation for column index
xi = round((m-patch-1)*xi)+1;       % Generate integer row index from real rand
yi = round((n-patch-1)*yi)+1;       % Generate integer col index from real rand
xyi = (yi-1)*m+xi;                  % Combine row and col indices into one index
xyi_unique = unique(xyi);           % Discovering the number of unique indices
xit = xyi_unique-floor(xyi_unique/m)*m; % Generate row index from unique index
yit = ceil(xyi_unique/m);               % Generate col index from unique index
nap = length(xyi_unique);               % Actual number of unique patches
% Displaying the actual number of unique patches
disp(["The actual number of unique patches is ", num2str(nap)]);
%-------------------------------------------------------------------------------
% Computing correlation matrix C
C = zeros(patch*patch, patch*patch);  % Initialize correlation matrix C
% For loop creates correlation matrix C from outer product of patch vectors
for i = 1 : nap,
  temp_patch = xr(xit(i):xit(i)+patch-1,yit(i):yit(i)+patch-1); % Indexing
  temp_patch = temp_patch(:); % Reshaping patch into a vector
  C = C + temp_patch*temp_patch'; % Outer product
end
%-------------------------------------------------------------------------------
% Eigenvalues and Eigenvectors of correlation matrix C
[e,l] = eig(C); % Computing eigenvectors and eigenvalues of C
[lsort,isort] = sort(diag(l),'descend'); % Sorting the eigenvalues (descending)
esort = e(:,isort); % Reordering the eigenvectors according to prev. sort
%-------------------------------------------------------------------------------
% Image I/O
imgrid = sqrt(number_eig); % Number of columns of eigenvector images
pa_sc = patch + scratch;   % Patch size plus scratch area for final display
imfinal = zeros(imgrid*(pa_sc),imgrid*(pa_sc)); % Initialize final disp. image
%-------------------------------------------------------------------------------
% Double for loop to create composite image comprising eigenvector images
for k = 1 : imgrid,
  for l = 1 : imgrid,
%   Forming final display image from eigenvector images
    imfinal((k-1)*pa_sc+1:(k-1)*pa_sc+patch,(l-1)*pa_sc+1:(l-1)*pa_sc+patch) = reshape(esort(:,(k-1)*imgrid+l),patch,patch);
  end
end
%-------------------------------------------------------------------------------
% Final Display
figure(1);
imagesc(imfinal); % Scaling final image for display 
colormap('gray'); % Setting colormap to grayscale
axis('image');    % Setting aspect ratio of image to square pixels
axis('off');      % Switching off additional axis information



