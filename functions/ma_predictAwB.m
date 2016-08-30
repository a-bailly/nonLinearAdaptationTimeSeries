% function [] = ma_predictAwB(domA, lbA, domB, lbB, N)
%
% Inputs:
%	domA:	Time Series (LTS) from Domain A
%	lbA:	Corresponding Labels for Time Series from Domain A
%	domB:	Time Series (LTS) from Domain B
%	lbB:	Corresponding Labels for Time Series from Domain B
%	Na:		Number of TS par class used as labeled example for Domain A
%
% Output:
%	-
%
% Adeline Bailly - 2016
% adeline.bailly@univ-rennes2.fr

function [] = ma_predictAwB(domA, lbA, domB, lbB, N)

%% Test

set(gcf,'PaperUnits','centimeters');
set(gcf, 'PaperType','A4');
orient landscape;

clear fig

%% division in train/test for each domain

r = 1;

[trA, lbtrA, tsA, lbtsA, ~] = ppc(domA, lbA, N, r);

domxpA = tsA(1:2:end,:)';
lbxpA = lbtsA(1:2:end,:);
tsA = tsA(2:2:end,:)';
lbtsA = lbtsA(2:2:end,:);

trB = domB(1:2:end,:)';
lbtrB = lbB(1:2:end,:);
tsB = domB(2:2:end,:)';
lbtsB = lbB(2:2:end,:);

trA = trA';

cl_rates = zeros(3,3);


%% Dom. A w. B
obj = fitcdiscr(trA', lbtrA);

label = predict(obj, tsA');
cl_rates(1,1) = sum((lbtsA == label))/numel(label);
label = predict(obj, domxpA');
cl_rates(2,1) = sum((lbxpA == label))/numel(label);
label = predict(obj, [tsA'; domxpA']);
cl_rates(3,1) = sum(([lbtsA; lbxpA] == label))/numel(label);

if (size(trA,1) == size(trB,1))
	obj = fitcdiscr(trB', lbtrB);

	label = predict(obj, tsA');
	cl_rates(1,2) = sum((lbtsA == label))/numel(label);
	label = predict(obj, domxpA');
	cl_rates(2,2) = sum((lbxpA == label))/numel(label);
	label = predict(obj, [tsA'; domxpA']);
	cl_rates(3,2) = sum(([lbtsA; lbxpA] == label))/numel(label);
	
	obj = fitcdiscr([trB'; tsB'], [lbtrB; lbtsB]);

	label = predict(obj, tsA');
	cl_rates(1,3) = sum((lbtsA == label))/numel(label);
	label = predict(obj, domxpA');
	cl_rates(2,3) = sum((lbxpA == label))/numel(label);
	label = predict(obj, [tsA'; domxpA']);
	cl_rates(3,3) = sum(([lbtsA; lbxpA] == label))/numel(label);
else			
	% Linear interpolation to fit length TS
%	trbisA    = interp1(linspace(0,1,size(trA,1)), trA,    linspace(0,1,size(trB,1)));
	tsbisA    = interp1(linspace(0,1,size(trA,1)), tsA,    linspace(0,1,size(trB,1)));
	domxpbisA = interp1(linspace(0,1,size(trA,1)), domxpA, linspace(0,1,size(trB,1)));
	obj = fitcdiscr(trB', lbtrB);

	label = predict(obj, tsbisA');
	cl_rates(1,2) = sum((lbtsA == label))/numel(label);
	label = predict(obj, domxpbisA');
	cl_rates(2,2) = sum((lbxpA == label))/numel(label);
	label = predict(obj, [tsbisA'; domxpbisA']);
	cl_rates(3,2) = sum(([lbtsA; lbxpA] == label))/numel(label);
	
	obj = fitcdiscr([trB'; tsB'], [lbtrB; lbtsB]);

	label = predict(obj, tsbisA');
	cl_rates(1,3) = sum((lbtsA == label))/numel(label);
	label = predict(obj, domxpbisA');
	cl_rates(2,3) = sum((lbxpA == label))/numel(label);
	label = predict(obj, [tsbisA'; domxpbisA']);
	cl_rates(3,3) = sum(([lbtsA; lbxpA] == label))/numel(label);
end

% disp(cl_rates(:)')

%%

options.graph.nn = 5;
options.mu = 1.;
options.fig = 1;
options.d = 10;

%% SSMA

domxpB = [];
lbxpB = [];
[SSA_u_l, SSA_v_l, SSA_uv_l, SSB_u_l, SSB_v_l, SSB_uv_l, SSAB_u_l, SSAB_v_l, SSAB_uv_l] = ssma_predictAwB(trA, lbtrA, tsA, lbtsA, domxpA, lbxpA, trB, lbtrB, tsB, lbtsB, options);
	
if (0 == 1)
	disp('')
	disp('[- SSMA - Classification rate -]'),

	disp('unlab(A) u. lab(A)'), disp(SSA_u_l')
	disp('val(A) u. lab(A)'), disp(SSA_v_l')
	disp('unlab+val(A) u. lab(A)'), disp(SSA_uv_l')
	disp('unlab(A) u. lab(B)'), disp(SSB_u_l')
	disp('val(A) u. lab(B)'), disp(SSB_v_l')
	disp('unlab+val(A) u. lab(B)'), disp(SSB_uv_l')
	disp('unlab(A) u. dom(B)'), disp(SSAB_u_l')
	disp('val(A) u. dom(B)'), disp(SSAB_v_l')
	disp('unlab+val(A) u. dom(B)'), disp(SSAB_uv_l')
end

% disp([max(SSA_u_l), max(SSA_v_l), max(SSA_uv_l), max(SSB_u_l), max(SSB_v_l), max(SSB_uv_l), max(SSAB_u_l), max(SSAB_v_l), max(SSAB_uv_l)])

%disp([mean(A_u_l(3:end)), mean(A_v_l(3:end)), mean(A_uv_l(3:end)), mean(B_u_l(3:end)), mean(B_v_l(3:end)), mean(B_uv_l(3:end)), mean(AB_u_l(3:end)), mean(AB_v_l(3:end)), mean(AB_uv_l(3:end))])

%% KEMA

% domxpB = domB(1,:);
% lbxpB = lbB(1,:);
[A_u_l, A_v_l, A_uv_l, B_u_l, B_v_l, B_uv_l, AB_u_l, AB_v_l, AB_uv_l] = kema_predictAwB(trA, lbtrA, tsA, lbtsA, domxpA, lbxpA, trB, lbtrB, tsB, lbtsB, options);

if (0 == 1)
	disp('')
	disp('[- KEMA - Classification rate -]'),

	disp('unlab(A) u. lab(A)'), disp(A_u_l')
	disp('val(A) u. lab(A)'), disp(A_v_l')
	disp('unlab+val(A) u. lab(A)'), disp(A_uv_l')
	disp('unlab(A) u. lab(B)'), disp(B_u_l')
	disp('val(A) u. lab(B)'), disp(B_v_l')
	disp('unlab+val(A) u. lab(B)'), disp(B_uv_l')
	disp('unlab(A) u. dom(B)'), disp(AB_u_l')
	disp('val(A) u. dom(B)'), disp(AB_v_l')
	disp('unlab+val(A) u. dom(B)'), disp(AB_uv_l')
end

% disp([max(A_u_l), max(A_v_l), max(A_uv_l), max(B_u_l), max(B_v_l), max(B_uv_l), max(AB_u_l), max(AB_v_l), max(AB_uv_l)])

if(options.fig == 1)
	t = num2cell([0, options.d, 0, 1]);
	[xmin, xmax, ymin, ymax] = deal(t{:});
	
	figur = gcf;
	set(gcf,'PaperUnits','centimeters');
	set(gcf, 'PaperType','A4');
	orient landscape;
	
	v = (1:options.d);
	
	bs_rates = repmat(cl_rates(1,1), 1,options.d);
	subplot(3, 3, 1);
	scatter(v, SSA_u_l, 'o'), hold on,
	scatter(v, A_u_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'), 
	legend('SSMA', 'KEMA', 'DAC', 'Location', 'South');
	ylabel('unlab(A) u. lab(A)'),
	grid on,
	axis([xmin xmax ymin ymax]); 
		
	bs_rates = repmat(cl_rates(2,1), 1,options.d);
	subplot(3, 3, 4);
	scatter(v, SSA_v_l, 'o'), hold on,
	scatter(v, A_v_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	ylabel('val(A) u. lab(A)'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(3,1), 1,options.d);
	subplot(3, 3, 7);
	scatter(v, SSA_uv_l, 'o'), hold on,
	scatter(v, A_uv_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	ylabel('unlab+val(A) u. lab(A)'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(1,2), 1,options.d);
	subplot(3, 3, 2);
	scatter(v, SSB_u_l, 'o'), hold on,
	scatter(v, B_u_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	ylabel('unlab(A) u. half(B)'),
	grid on,
	axis([xmin xmax ymin ymax]);
 	
 	bs_rates = repmat(cl_rates(2,2), 1,options.d);
 	subplot(3, 3, 5);
 	scatter(v, SSB_v_l, 'o'), hold on,
 	scatter(v, B_v_l, 'x'),  hold on,
 	scatter(v, bs_rates, '+'),
	ylabel('val(A) u. half(B)'),
 	grid on,
 	axis([xmin xmax ymin ymax]);
 	
 	bs_rates = repmat(cl_rates(3,2), 1,options.d);
 	subplot(3, 3, 8);
 	scatter(v, SSB_uv_l, 'o'), hold on,
 	scatter(v, B_uv_l, 'x'),  hold on,
 	scatter(v, bs_rates, '+'),
	ylabel('unlab+val(A) u. half(B)'),
 	grid on,
 	axis([xmin xmax ymin ymax]);
 	
 	bs_rates = repmat(cl_rates(1,3), 1,options.d);
 	subplot(3, 3, 3);
 	scatter(v, SSAB_u_l, 'o'), hold on,
 	scatter(v, AB_u_l, 'x'),  hold on,
 	scatter(v, bs_rates, '+'),
	ylabel('unlab(A) u. B'),
 	grid on,
 	axis([xmin xmax ymin ymax]);
 	
 	bs_rates = repmat(cl_rates(2,3), 1,options.d);
 	subplot(3, 3, 6);
 	scatter(v, SSAB_v_l, 'o'), hold on,
 	scatter(v, AB_v_l, 'x'),  hold on,
 	scatter(v, bs_rates, '+'), 
	ylabel('val(A) u. B'),
 	grid on,
 	axis([xmin xmax ymin ymax]);
 	
 	bs_rates = repmat(cl_rates(3,3), 1,options.d);
 	subplot(3, 3, 9);
 	scatter(v, SSAB_uv_l, 'o'), hold on,
 	scatter(v, AB_uv_l, 'x'),  hold on,
 	scatter(v, bs_rates, '+'),
	ylabel('unlab+val(A) u. B'),
 	grid on,
 	axis([xmin xmax ymin ymax]);
	
	saveas(figur, 'classif_ratesAwB', 'pdf');
	clear figur;
	
	%disp('Press enter to continue'); pause
end

%disp([mean(A_u_l(3:end)), mean(A_v_l(3:end)), mean(A_uv_l(3:end)), mean(B_u_l(3:end)), mean(B_v_l(3:end)), mean(B_uv_l(3:end)), mean(AB_u_l(3:end)), mean(AB_v_l(3:end)), mean(AB_uv_l(3:end))])

%clear all
