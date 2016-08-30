% function [] = ma_predict(domA, lbA, domB, lbB, N)
%
% Inputs:
%	domA:	Time Series (LTS) from Domain A
%	lbA:	Corresponding Labels for Time Series from Domain A
%	domB:	Time Series (LTS) from Domain B
%	lbB:	Corresponding Labels for Time Series from Domain B
%	N:		Number of TS par class used as labeled example
%
% Output:
%	-
%
% Adeline Bailly - 2016
% adeline.bailly@univ-rennes2.fr

function [] = ma_predict(domA, lbA, domB, lbB, N)

%% Test

set(gcf,'PaperUnits','centimeters');
set(gcf, 'PaperType','A4');
orient landscape;

clear fig
% disp('Please press enter to continue'); pause

%% division in train/test for each domain

r = 1;

[trA, lbtrA, tsA, lbtsA, ~] = ppc(domA, lbA, N, r);
[trB, lbtrB, tsB, lbtsB, ~] = ppc(domB, lbB, N, r);

domxpA = tsA(1:2:end,:)';
domxpB = tsB(1:2:end,:)';
lbxpA = lbtsA(1:2:end,:);
lbxpB = lbtsB(1:2:end,:);

tsA = tsA(2:2:end,:);
tsB = tsB(2:2:end,:);
lbtsA = lbtsA(2:2:end,:);
lbtsB = lbtsB(2:2:end,:);

trA = trA'; trB = trB';
tsA = tsA'; tsB = tsB';

cl_rates = zeros(3,3);

%% Dom. A

obj = fitcdiscr(trA', lbtrA);

label = predict(obj, tsA');
cl_rates(1,1) = sum((lbtsA == label))/numel(label);

label = predict(obj, domxpA');
cl_rates(2,1) = sum((lbxpA == label))/numel(label);

label = predict(obj, [tsA'; domxpA']);
cl_rates(3,1) = sum(([lbtsA; lbxpA] == label))/numel(label);

%% Dom. B
obj = fitcdiscr(trB', lbtrB);

label = predict(obj, tsB');
cl_rates(1,2) = sum((lbtsB == label))/numel(label);

label = predict(obj, domxpB');
cl_rates(2,2) = sum((lbxpB == label))/numel(label);

label = predict(obj, [tsB'; domxpB']);
cl_rates(3,2) = sum(([lbtsB; lbxpB] == label))/numel(label);

%% Dom. A+B
if (size(trA,1) == size(trB,1))
	obj = fitcdiscr([trA'; trB'], [lbtrA; lbtrB]);

	label = predict(obj, [tsA'; tsB']);
	cl_rates(1,3) = sum(([lbtsA; lbtsB] == label))/numel(label);

	label = predict(obj, [domxpA'; domxpB']);
	cl_rates(2,3) = sum(([lbxpA; lbxpB] == label))/numel(label);

	label = predict(obj, [tsA'; tsB'; domxpA'; domxpB']);
	cl_rates(3,3) = sum(([lbtsA; lbtsB; lbxpA; lbxpB] == label))/numel(label);
else			
	% Linear interpolation to fit length TS
	trbisA    = interp1(linspace(0,1,size(trA,1)), trA,    linspace(0,1,size(trB,1)));
	tsbisA    = interp1(linspace(0,1,size(trA,1)), tsA,    linspace(0,1,size(trB,1)));
	domxpbisA = interp1(linspace(0,1,size(trA,1)), domxpA, linspace(0,1,size(trB,1)));
	
	obj = fitcdiscr([trbisA'; trB'], [lbtrA; lbtrB]);

	label = predict(obj, [tsbisA'; tsB']);
	cl_rates(1,3) = sum(([lbtsA; lbtsB] == label))/numel(label);

	label = predict(obj, [domxpbisA'; domxpB']);
	cl_rates(2,3) = sum(([lbxpA; lbxpB] == label))/numel(label);

	label = predict(obj, [tsbisA'; tsB'; domxpbisA'; domxpB']);
	cl_rates(3,3) = sum(([lbtsA; lbtsB; lbxpA; lbxpB] == label))/numel(label);
end
% pause
% return

% disp(cl_rates(:)')

%%

options.graph.nn = 5;
options.mu = 1.;
options.fig = 1;
options.d = 10;

%% SSMA

[SSA_u_l, SSA_v_l, SSA_uv_l, SSB_u_l, SSB_v_l, SSB_uv_l, SSAB_u_l, SSAB_v_l, SSAB_uv_l] = ssma_predict(trA, lbtrA, tsA, lbtsA, domxpA, lbxpA, trB, lbtrB, tsB, lbtsB, domxpB, lbxpB, options);

if (0 == 1)
	disp('')
	disp('[- SSMA - Classification rate -]'),

	disp('unlab(A) u. lab(A)'), disp(SSA_u_l')
	disp('val(A) u. lab(A)'), disp(SSA_v_l')
	disp('unlab+val(A) u. lab(A)'), disp(SSA_uv_l')
	disp('unlab(B) u. lab(B)'), disp(SSB_u_l')
	disp('val(B) u. lab(B)'), disp(SSB_v_l')
	disp('unlab+val(B) u. lab(B)'), disp(SSB_uv_l')
	disp('unlab(A+B) u. lab(A+B)'), disp(SSAB_u_l')
	disp('val(A+B) u. lab(A+B)'), disp(SSAB_v_l')
	disp('unlab+val(A+B) u. lab(A+B)'), disp(SSAB_uv_l')
end
% disp([max(SSA_u_l), max(SSA_v_l), max(SSA_uv_l), max(SSB_u_l), max(SSB_v_l), max(SSB_uv_l), max(SSAB_u_l), max(SSAB_v_l), max(SSAB_uv_l)])

%disp([mean(A_u_l(3:end)), mean(A_v_l(3:end)), mean(A_uv_l(3:end)), mean(B_u_l(3:end)), mean(B_v_l(3:end)), mean(B_uv_l(3:end)), mean(AB_u_l(3:end)), mean(AB_v_l(3:end)), mean(AB_uv_l(3:end))])

%% KEMA

[A_u_l, A_v_l, A_uv_l, B_u_l, B_v_l, B_uv_l, AB_u_l, AB_v_l, AB_uv_l] = kema_predict(trA, lbtrA, tsA, lbtsA, domxpA, lbxpA, trB, lbtrB, tsB, lbtsB, domxpB, lbxpB, options);
	
if (0 == 1)
	disp('')
	disp('[- KEMA - Classification rate -]'),

	disp('unlab(A) u. lab(A)'), disp(A_u_l')
	disp('val(A) u. lab(A)'), disp(A_v_l')
	disp('unlab+val(A) u. lab(A)'), disp(A_uv_l')
	disp('unlab(B) u. lab(B)'), disp(B_u_l')
	disp('val(B) u. lab(B)'), disp(B_v_l')
	disp('unlab+val(B) u. lab(B)'), disp(B_uv_l')
	disp('unlab(A+B) u. lab(A+B)'), disp(AB_u_l')
	disp('val(A+B) u. lab(A+B)'), disp(AB_v_l')
	disp('unlab+val(A+B) u. lab(A+B)'), disp(AB_uv_l')
end

% disp([max(A_u_l), max(A_v_l), max(A_uv_l), max(B_u_l), max(B_v_l), max(B_uv_l), max(AB_u_l), max(AB_v_l), max(AB_uv_l)])
% 
% disp('---')
% vvvv = (A_u_l * size(domA,1) + B_u_l * size(domB,1)) / ( size(domA,1) + size(domB,1));
% disp(vvvv)

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
	ylabel('unlab u. lab'),
	grid on,
	axis([xmin xmax ymin ymax]); 
		
	bs_rates = repmat(cl_rates(2,1), 1,options.d);
	subplot(3, 3, 4);
	scatter(v, SSA_v_l, 'o'), hold on,
	scatter(v, A_v_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	ylabel('val u. lab'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(3,1), 1,options.d);
	subplot(3, 3, 7);
	scatter(v, SSA_uv_l, 'o'), hold on,
	scatter(v, A_uv_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	legend('SSMA', 'KEMA', 'DAC', 'Location', 'North');
	ylabel('unlab+val u. lab'),
	xlabel('dom A'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(1,2), 1,options.d);
	subplot(3, 3, 2);
	scatter(v, SSB_u_l, 'o'), hold on,
	scatter(v, B_u_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(2,2), 1,options.d);
	subplot(3, 3, 5);
	scatter(v, SSB_v_l, 'o'), hold on,
	scatter(v, B_v_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(3,2), 1,options.d);
	subplot(3, 3, 8);
	scatter(v, SSB_uv_l, 'o'), hold on,
	scatter(v, B_uv_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
	xlabel('dom B'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(1,3), 1,options.d);
	subplot(3, 3, 3);
	scatter(v, SSAB_u_l, 'o'), hold on,
	scatter(v, AB_u_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
% 	scatter(v, (A_u_l * size(domA,1) + B_u_l * size(domB,1)) / ( size(domA,1) + size(domB,1)), 'gp'),  hold on,
% 	scatter(v, (SSA_u_l * size(domA,1) + SSB_u_l * size(domB,1)) / ( size(domA,1) + size(domB,1)), 'mp'),  hold on,
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(2,3), 1,options.d);
	subplot(3, 3, 6);
	scatter(v, SSAB_v_l, 'o'), hold on,
	scatter(v, AB_v_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
% 	scatter(v, (A_v_l * size(domA,1) + B_v_l * size(domB,1)) / ( size(domA,1) + size(domB,1)), 'gp'),  hold on,
% 	scatter(v, (SSA_v_l * size(domA,1) + SSB_v_l * size(domB,1)) / ( size(domA,1) + size(domB,1)), 'mp'),  hold on,
	grid on,
	axis([xmin xmax ymin ymax]);
	
	bs_rates = repmat(cl_rates(3,3), 1,options.d);
	subplot(3, 3, 9);
	scatter(v, SSAB_uv_l, 'o'), hold on,
	scatter(v, AB_uv_l, 'x'),  hold on,
	scatter(v, bs_rates, '+'),
% 	scatter(v, (A_uv_l * size(domA,1) + B_uv_l * size(domB,1)) / ( size(domA,1) + size(domB,1)), 'gp'),  hold on,
% 	scatter(v, (SSA_uv_l * size(domA,1) + SSB_uv_l * size(domB,1)) / ( size(domA,1) + size(domB,1)), 'mp'),  hold on,
	xlabel('dom AB'),
	grid on,
	axis([xmin xmax ymin ymax]);
	
	saveas(figur, 'classif_rates', 'pdf');
	clear figur;
	
	%disp('Press enter to continue'); pause
end

%disp([mean(A_u_l(3:end)), mean(A_v_l(3:end)), mean(A_uv_l(3:end)), mean(B_u_l(3:end)), mean(B_v_l(3:end)), mean(B_uv_l(3:end)), mean(AB_u_l(3:end)), mean(AB_v_l(3:end)), mean(AB_uv_l(3:end))])

%clear all
