% function [KE_A_u, KE_A_t, KE_A_ut, KE_B_u, KE_B_t, KE_B_ut] = kema_predict(trainA_labeled, labelsA, trainA_unlabeled, unlabelsA, testA, tlabelsA, trainB_labeled, labelsB, trainB_unlabeled, unlabelsB, testB, tlabelsB, options)
%
% Inputs:
%	trainA_labeled:		Labeled Time Series (LTS) from Domain A
%	labelsA:			Corresponding Labels for Labeled Time Series from Domain A
%	trainA_unlabeled:	Unlabeled Time Series (UTS) from Domain A
%	unlabelsA:			Corresponding Labels for Unlabeled Time Series from Domain A
%	testA:				Time Series from Domain A for Validation (VTS)
%	tlabelsA:			Corresponding Labels for Validation Time Series from Domain A
%	trainB_labeled:		Labeled Time Series from Domain B
%	labelsB:			Corresponding Labels for Time Series from Domain B
%	trainB_unlabeled:	Unlabeled Time Series from Domain B
%	unlabelsB:			Corresponding Labels for Unlabeled Time Series from Domain B
%	testB:				Time Series from Domain B for Validation
%	tlabelsB:			Corresponding Labels for Validation Time Series from Domain B
%	options:			Various options
%
% Outputs:
%	KE_A_u:		Classification rate of unlabeled in Domain F using labeled from domain AB
%	KE_A_t:		Classification rate of test in Domain F using labeled from domain AB
%	KE_A_ut:	Classification rate of unlabeled+test in Domain F using labeled from domain AB
%	KE_B_u:		Classification rate of unlabeled in Domain F using labeled from domain AB
%	KE_B_t:		Classification rate of test in Domain F using labeled from domain AB
%	KE_B_ut:	Classification rate of unlabeled+test in Domain F using labeled from domain AB

%
% Adeline Bailly - 2016
% adeline.bailly@univ-rennes2.fr

function [KE_A_u, KE_A_t, KE_A_ut, KE_B_u, KE_B_t, KE_B_ut] = kema_predict(trainA_labeled, labelsA, trainA_unlabeled, unlabelsA, testA, tlabelsA, trainB_labeled, labelsB, trainB_unlabeled, unlabelsB, testB, tlabelsB, options)

%% KEMA

n_cl = numel(unique(labelsA));

labelsAu = zeros(size(trainA_unlabeled, 2), 1); 
labelsBu = zeros(size(trainB_unlabeled, 2), 1); 

% construct graph
G1 = buildKNNGraph([trainA_labeled, trainA_unlabeled]', options.graph.nn, 1); 
G2 = buildKNNGraph([trainB_labeled, trainB_unlabeled]', options.graph.nn, 1); 

W = blkdiag(G1, G2); W = double(full(W)); 
clear G*

% similarity matrices
Y = [labelsA; labelsAu; labelsB; labelsBu]; 

Ws = repmat(Y, 1, length(Y)) == repmat(Y, 1, length(Y))'; 
Ws(Y == 0, :) = 0; Ws(:, Y == 0) = 0; Ws = double(Ws); 

Wd = repmat(Y, 1, length(Y)) ~= repmat(Y, 1, length(Y))'; 
Wd(Y == 0, :) = 0; Wd(:, Y == 0) = 0; Wd = double(Wd); 

Sw  = sum(sum(W)); 
Sws = sum(sum(Ws)); Ws = Ws/Sws*Sw; 
Swd = sum(sum(Wd)); Wd = Wd/Swd*Sw; 

clear Sw*

% dissimilarity matrices
D  = sum(W, 2); 
Ds = sum(Ws, 2); 
Dd = sum(Wd, 2); 

% graph laplacian matrix
L  = diag(D) - W; 
Ls = diag(Ds) - Ws; 
Ld = diag(Dd) - Wd; 

% 
mtxA = options.mu*L + Ls; mtxB = Ld; 

% RBF kernels
sigma1 = mean(pdist(trainA_labeled')); 
K1 = kernelmatrix('rbf', [trainA_labeled, trainA_unlabeled], [trainA_labeled, trainA_unlabeled], sigma1); 
sigma2 = mean(pdist(trainB_labeled')); 
K2 = kernelmatrix('rbf', [trainB_labeled, trainB_unlabeled], [trainB_labeled, trainB_unlabeled], sigma2); 

K = blkdiag(K1, K2); 

KT1 = kernelmatrix('rbf', [trainA_labeled, trainA_unlabeled], testA, sigma1); 
KT2 = kernelmatrix('rbf', [trainB_labeled, trainB_unlabeled], testB, sigma1); 

[V, lambda] = gen_eig(K*mtxA*K, K*mtxB*K, 'LM'); 

[~, j] = sort(diag(lambda)); 
V = V(:, j); 

lenA = size(trainA_labeled, 2) + size(trainA_unlabeled, 2); 

%% rotation
E1 = V(1:lenA, :); 
E2 = V(lenA+1:end, :); 

sourceXpInv = (E1'*K1*-1)'; 
sourceXp = (E1'*K1)'; 
targetXp = (E2'*K2)'; 

sourceXpInv = zscore(sourceXpInv); 
sourceXp = zscore(sourceXp); 
targetXp = zscore(targetXp); 

ErrRec = zeros(numel(unique(labelsA)), size(V, 2)); 
ErrRecInv = zeros(numel(unique(labelsA)), size(V, 2)); 

m1 = zeros(numel(unique(labelsA)), size(V, 2)); 
m1inv = zeros(numel(unique(labelsA)), size(V, 2)); 
m2 = zeros(numel(unique(labelsA)), size(V, 2)); 

cls = unique(labelsA); 

for j = 1:size(V, 2)
	for i = 1:numel(cls)
		m1inv(i, j) = mean(sourceXpInv([labelsA; labelsAu]==cls(i), j)); 
		m1(i, j) = mean(sourceXp([labelsA; labelsAu]==cls(i), j)); 
		m2(i, j) = mean(targetXp([labelsB; labelsBu]==cls(i), j)); 

		ErrRec(i, j) = sqrt((mean(sourceXp([labelsA; labelsAu]==cls(i), j))-mean(targetXp([labelsB; labelsBu]==cls(i), j))).^2); 
		ErrRecInv(i, j) = sqrt((mean(sourceXpInv([labelsA; labelsAu]==cls(i), j))-mean(targetXp([labelsB; labelsBu]==cls(i), j))).^2); 
	end
end

mean(ErrRec); 
mean(ErrRecInv); 

Sc = max(ErrRec)>max(ErrRecInv); 
V(1:lenA, Sc) = V(1:lenA, Sc)*-1; 

clear cls E* i j
%% -- rotation

%clear L* W*

options.d = min(options.d, size(trainA_labeled,2)+ size(trainB_labeled,2)- n_cl);
KE_A_u  = zeros(options.d, 1);
KE_A_t   = zeros(options.d, 1);
KE_A_ut  = zeros(options.d, 1);
KE_B_u   = zeros(options.d, 1);
KE_B_t   = zeros(options.d, 1);
KE_B_ut  = zeros(options.d, 1);

set(gcf,'PaperUnits','centimeters');
set(gcf, 'PaperType','A4');
orient landscape;

for dd = 1:options.d
	vAF = V(1:lenA, 1:dd); 
	vBF = V(lenA+1:end, 1:dd); 
	
	% Projection to latent space
	AtoF = vAF' * K1; 
	BtoF = vBF' * K2; 
	
	mA = mean(AtoF, 2)'; sA = std(AtoF, 0, 2)'; 
	mB = mean(BtoF, 2)'; sB = std(BtoF, 0, 2)'; 
	AtoF = ((AtoF' - repmat(mA, size(AtoF, 2), 1)) ./ repmat(sA, size(AtoF, 2), 1))'; 
	BtoF = ((BtoF' - repmat(mB, size(BtoF, 2), 1)) ./ repmat(sB, size(BtoF, 2), 1))'; 
	
	% Projection of xp data
	AxptoF = vAF' * KT1; 
	mA = mean(AxptoF, 2)'; sA = std(AxptoF, 0, 2)'; 
	AxptoF = ((AxptoF' - repmat(mA, size(AxptoF, 2), 1)) ./ repmat(sA, size(AxptoF, 2), 1))'; 
	
	BxptoF = vBF' * KT2; 
	mB = mean(BxptoF, 2)'; sB = std(BxptoF, 0, 2)'; 
	BxptoF = ((BxptoF' - repmat(mB, size(BxptoF, 2), 1)) ./ repmat(sB, size(BxptoF, 2), 1))'; 
	
	% Prediction
	AtstoF = AtoF(:, size(labelsA, 1)+1:end); 
	AtoF   = AtoF(:, 1:size(labelsA, 1));
	BtstoF = BtoF(:, size(labelsB, 1)+1:end); 
	BtoF   = BtoF(:, 1:size(labelsB, 1));
		
	obj = fitcdiscr([AtoF'; BtoF'], [labelsA; labelsB]);
	
 	predictA = predict(obj, AtstoF');
	KE_A_u(dd) = sum((unlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, AxptoF');
	KE_A_t(dd) = sum((tlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, [AtstoF'; AxptoF']);
	KE_A_ut(dd) = sum(([unlabelsA; tlabelsA] == predictA))/numel(predictA);
	
 	predictB = predict(obj, BtstoF');
	KE_B_u(dd) = sum((unlabelsB == predictB))/numel(predictB);
 	predictB = predict(obj, BxptoF');
	KE_B_t(dd) = sum((tlabelsB == predictB))/numel(predictB);
 	predictB = predict(obj, [BtstoF'; BxptoF']);
	KE_B_ut(dd) = sum(([unlabelsB; tlabelsB] == predictB))/numel(predictB);
end

if(options.fig == 1)
	
	fontname = 'Times';
	fontsize = 20;
	fontunits = 'points';
	set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
		'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
		'DefaultLineLineWidth',1,'DefaultLineMarkerSize',2,'DefaultLineColor',[0 0 0]);
		
% 	t = num2cell([-4, 4, -4, 4]);
% 	[xmin, xmax, ymin, ymax] = deal(t{:});
	xmin = min( [min(AtoF(1, :)), min(AtstoF(1,:)), min(AxptoF(1,:))]) -.5;
	xmax = max( [max(AtoF(1, :)), max(AtstoF(1,:)), max(AxptoF(1,:))]) +.5;
	ymin = min( [min(AtoF(2, :)), min(AtstoF(2,:)), min(AxptoF(2,:))]) -.5;
	ymax = max( [max(AtoF(2, :)), max(AtstoF(2,:)), max(AxptoF(2,:))]) +.5;
	figur = gcf;

	szpoint = 40;

	subplot(2, 3, 1);
	hh = gca;
	llabels = unique(labelsA);
	colormap(jet(n_cl));
	markers = ['o', 'd', 'v', 's', '>', '<', 'h', 'p'];
	for ii=1:n_cl
		k = find(labelsA==llabels(ii));
		scatter(AtoF(1, k)', AtoF(2, k)', szpoint, markers(ii), 'filled'), hold on;
	end
	hold off;	xlabel('dim. 1', 'FontSize', 16),	ylabel('dim. 2', 'FontSize', 16),
	grid off,	set(gca,'xtick',[],'ytick',[]);		axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 2);
%	scatter(AtstoF(1, :)', AtstoF(2, :)', szpoint, unlabelsA, '.'), 
	colormap(jet(n_cl)),
	for ii=1:n_cl
		k = find(unlabelsA==llabels(ii));
		scatter(AtstoF(1, k)', AtstoF(2, k)', szpoint, markers(ii), 'filled'), hold on;
	end
	hold off;	xlabel('dim. 1', 'FontSize', 16),	ylabel('dim. 2', 'FontSize', 16),
	grid off,	set(gca,'xtick',[],'ytick',[]);		axis([xmin xmax ymin ymax]);
	
	subplot(2, 3, 3);
%	scatter(AxptoF(1, :)', AxptoF(2, :)', szpoint, tlabelsA, '.'), 
	for ii=1:n_cl
		k = find(tlabelsA==llabels(ii));
		scatter(AxptoF(1, k)', AxptoF(2, k)', szpoint, markers(ii), 'filled'), hold on;
	end
	hold off;	xlabel('dim. 1', 'FontSize', 16),	ylabel('dim. 2', 'FontSize', 16),
	grid off,	set(gca,'xtick',[],'ytick',[]);		axis([xmin xmax ymin ymax]);
	
	subplot(2, 3, 4);
%	scatter(BtoF(1, :)', BtoF(2, :)', szpoint, labelsB, '.'), 
	for ii=1:n_cl
		k = find(labelsB==llabels(ii));
		scatter(BtoF(1, k)', BtoF(2, k)', szpoint, markers(ii), 'filled'), hold on;
	end
	hold off;	xlabel('dim. 1', 'FontSize', 16),	ylabel('dim. 2', 'FontSize', 16),
	grid off,	set(gca,'xtick',[],'ytick',[]);		axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 5);
%	scatter(BtstoF(1, :)', BtstoF(2, :)', szpoint, unlabelsB, '.'), 
	for ii=1:n_cl
		k = find(labelsB==llabels(ii));
		scatter(BtstoF(1, k)', BtstoF(2, k)', szpoint, markers(ii), 'filled'), hold on;
	end
	hold off;	xlabel('dim. 1', 'FontSize', 16),	ylabel('dim. 2', 'FontSize', 16),
	grid off,	set(gca,'xtick',[],'ytick',[]);		axis([xmin xmax ymin ymax]); 

	if (size(testB,1) > 0)
		subplot(2, 3, 6);
%		scatter(BxptoF(1, :)', BxptoF(2, :)', szpoint, tlabelsB, '.'), 
		for ii=1:n_cl
			k = find(labelsB==llabels(ii));
			scatter(BxptoF(1, k)', BxptoF(2, k)', szpoint, markers(ii), 'filled'), hold on;
		end

		hold off;	xlabel('dim. 1', 'FontSize', 16),	ylabel('dim. 2', 'FontSize', 16),
		grid off,	set(gca,'xtick',[],'ytick',[]);		axis([xmin xmax ymin ymax]); 
	end

	saveas(figur, 'kema', 'pdf')
	%disp('Press enter to continue'); pause
end

clear options r s* m* d dd D*
clear *toF L* Sc V W* Y Z fig predictA t targetXp
