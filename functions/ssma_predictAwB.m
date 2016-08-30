% function [A_u_l, A_v_l, A_uv_l, B_u_l, B_v_l, B_uv_l, AB_u_l, AB_v_l, AB_uv_l] = ssma_predict_AwB(trainA_labeled, labelsA, trainA_unlabeled, unlabelsA, testA, tlabelsA, trainB_labeled, labelsB, trainB_unlabeled, unlabelsB, options)
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
%	options:			Various options
%
% Outputs:
%	A_u_l:		Classification rate of 
%	A_v_l:		Classification rate of 
%	A_uv_l:		Classification rate of 
%	B_u_l:		Classification rate of 
%	B_v_l:		Classification rate of 
%	B_uv_l:		Classification rate of 
%	AB_u_l:		Classification rate of 
%	AB_v_l:		Classification rate of 
%	AB_uv_l:	Classification rate of 
%
%
% Adeline Bailly - 2016
% adeline.bailly@univ-rennes2.fr

function [A_u_l, A_v_l, A_uv_l, B_u_l, B_v_l, B_uv_l, AB_u_l, AB_v_l, AB_uv_l] = ssma_predictAwB(trainA_labeled, labelsA, trainA_unlabeled, unlabelsA, testA, tlabelsA, trainB_labeled, labelsB, trainB_unlabeled, unlabelsB, options)

%% SSMA

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
Z = blkdiag([trainA_labeled, trainA_unlabeled], [trainB_labeled, trainB_unlabeled]);

[V, ~] = eigs(Z*mtxA*Z', Z*mtxB*Z', options.d, 'sm'); %small magnitude

lenA = size(trainA_labeled, 1);
% lenB = size(trainB_labeled, 1);

%% rotation
E1 = V(1:lenA,:);
E2 = V(lenA+1:end,:);

sourceXpInv = (E1'*trainA_labeled*-1)';
sourceXp = (E1'*trainA_labeled)';
targetXp = (E2'*trainB_labeled)';

sourceXpInv = zscore(sourceXpInv);
sourceXp = zscore(sourceXp);
targetXp = zscore(targetXp);

ErrRec = zeros(numel(unique(labelsA)),size(V,2));
ErrRecInv = zeros(numel(unique(labelsA)),size(V,2));

m1 = zeros(numel(unique(labelsA)),size(V,2));
m1inv = zeros(numel(unique(labelsA)),size(V,2));
m2 = zeros(numel(unique(labelsA)),size(V,2));

cls = unique(labelsA);

for j = 1:size(V,2)
	for i = 1:numel(cls)
		m1inv(i,j) = mean(sourceXpInv([labelsA;labelsAu]==cls(i),j));
		m1(i,j) = mean(sourceXp([labelsA;labelsAu]==cls(i),j));
		m2(i,j) = mean(targetXp([labelsB;labelsBu]==cls(i),j));

		ErrRec(i,j) = sqrt((mean(sourceXp([labelsA;labelsAu]==cls(i),j))-mean(targetXp([labelsB;labelsBu]==cls(i),j))).^2);
		ErrRecInv(i,j) = sqrt((mean(sourceXpInv([labelsA;labelsAu]==cls(i),j))-mean(targetXp([labelsB;labelsBu]==cls(i),j))).^2);
	end
end

mean(ErrRec);
mean(ErrRecInv);

Sc = max(ErrRec)>max(ErrRecInv);
V(1:lenA,Sc) = V(1:lenA,Sc)*-1;

clear cls E* i j
%% -- rotation

%clear L* W*

options.d = min(options.d, size(trainA_labeled,2)+ size(trainB_labeled,2)- n_cl);
A_u_l  = zeros(options.d, 1);
A_v_l   = zeros(options.d, 1);
A_uv_l  = zeros(options.d, 1);
B_u_l   = zeros(options.d, 1);
B_v_l   = zeros(options.d, 1);
B_uv_l  = zeros(options.d, 1);
AB_u_l  = zeros(options.d, 1);
AB_v_l  = zeros(options.d, 1);
AB_uv_l = zeros(options.d, 1);

set(gcf,'PaperUnits','centimeters');
set(gcf, 'PaperType','A4');
orient landscape;

%halfd = options.d / 2;

for dd = 1:options.d
	vAF = V(1:lenA, 1:dd); 
	vBF = V(lenA+1:end, 1:dd); 
	
	% Projection to latent space
	AtoF = vAF' * trainA_labeled;
	BtoF = vBF' * trainB_labeled;
	
	mA = mean(AtoF, 2)'; sA = std(AtoF, 0, 2)';
	mB = mean(BtoF, 2)'; sB = std(BtoF, 0, 2)';
	AtoF = ((AtoF' - repmat(mA, size(AtoF, 2), 1)) ./ repmat(sA, size(AtoF, 2), 1))';
	BtoF = ((BtoF' - repmat(mB, size(BtoF, 2), 1)) ./ repmat(sB, size(BtoF, 2), 1))'; 
	
 	% Projection of unlabeled data
	AtstoF = vAF' * trainA_unlabeled;
	BtstoF = vBF' * trainB_unlabeled;
	
	mA = mean(AtstoF, 2)'; sA = std(AtstoF, 0, 2)';
	mB = mean(BtstoF, 2)'; sB = std(BtstoF, 0, 2)';
	AtstoF = ((AtstoF' - repmat(mA, size(AtstoF, 2), 1)) ./ repmat(sA, size(AtstoF, 2), 1))';
	BtstoF = ((BtstoF' - repmat(mB, size(BtstoF, 2), 1)) ./ repmat(sB, size(BtstoF, 2), 1))';
	
	% Projection of xp data
	AxptoF = vAF' * testA;
	mA = mean(AxptoF, 2)'; sA = std(AxptoF, 0, 2)';
	AxptoF = ((AxptoF' - repmat(mA, size(AxptoF, 2), 1)) ./ repmat(sA, size(AxptoF, 2), 1))';
		
	obj = fitcdiscr(AtoF', labelsA);
	
 	predictA = predict(obj, AtstoF');
	A_u_l(dd) = sum((unlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, AxptoF');
	A_v_l(dd) = sum((tlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, [AtstoF'; AxptoF']);
	A_uv_l(dd) = sum(([unlabelsA; tlabelsA] == predictA))/numel(predictA);
	
	obj = fitcdiscr(BtoF', labelsB);
	
 	predictA = predict(obj, AtstoF');
	B_u_l(dd) = sum((unlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, AxptoF');
	B_v_l(dd) = sum((tlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, [AtstoF'; AxptoF']);
	B_uv_l(dd) = sum(([unlabelsA; tlabelsA] == predictA))/numel(predictA);
	
	obj = fitcdiscr([BtoF'; BtstoF'], [labelsB; unlabelsB]);
	
 	predictA = predict(obj, AtstoF');
	AB_u_l(dd) = sum((unlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, AxptoF');
	AB_v_l(dd) = sum((tlabelsA == predictA))/numel(predictA);
 	predictA = predict(obj, [AtstoF'; AxptoF']);
	AB_uv_l(dd) = sum(([unlabelsA; tlabelsA] == predictA))/numel(predictA);
end

if(options.fig == 1)
	t = num2cell([-4, 4, -4, 4]);
	[xmin, xmax, ymin, ymax] = deal(t{:});
	
	figur = gcf;
	set(gcf,'PaperUnits','centimeters');
	set(gcf, 'PaperType','A4');
	orient landscape;

	szpoint = 100;

	subplot(2, 3, 1);
	scatter(AtoF(1, :)', AtoF(2, :)', 100, labelsA, '.'), 
	colormap(jet(n_cl)),
	ylabel('dom. A to latent domain'),
	xlabel('labeled TS'),
	grid on,
	axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 2);
	scatter(AtstoF(1, :)', AtstoF(2, :)', szpoint, unlabelsA, '.'), 
	colormap(jet(n_cl)),
	xlabel('unlabeled TS'),
	grid on,
	axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 3);
	scatter(AxptoF(1, :)', AxptoF(2, :)', szpoint, tlabelsA, '.'), 
	colormap(jet(n_cl)),
	xlabel('validation TS'),
	grid on,
	axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 4);
	scatter(BtoF(1, :)', BtoF(2, :)', szpoint, labelsB, '.'), 
	colormap(jet(n_cl)),
	ylabel('dom. B to latent domain'),
	grid on,
	axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 5);
	scatter(BtstoF(1, :)', BtstoF(2, :)', szpoint, unlabelsB, '.'), 
	colormap(jet(n_cl)),
	grid on,
	axis([xmin xmax ymin ymax]); 
	
	subplot(2, 3, 6);
	grid on,
	axis([xmin xmax ymin ymax]); 
	
	saveas(figur, 'ssmaAwB', 'pdf')
	
	%disp('Press enter to continue'); pause
end

clear options r s* m* d dd D*
clear *toF L* Sc V W* Y Z fig predictA t targetXp
