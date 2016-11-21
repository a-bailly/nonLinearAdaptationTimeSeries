% function [] = run_xp(domA, lbA, domB, lbB, N)
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

function [] = run_xp(domA, lbA, domB, lbB, N)

%% Test

clear fig

r = 1;

[labA, lab_labA, unlA, lab_unlA, ~] = ppc(domA, lbA, N, r);
[labB, lab_labB, unlB, lab_unlB, ~] = ppc(domB, lbB, N, r);

testA = unlA(1:2:end,:)';
testB = unlB(1:2:end,:)';
lab_testA = lab_unlA(1:2:end,:);
lab_testB = lab_unlB(1:2:end,:);

unlA = unlA(2:2:end,:);
unlB = unlB(2:2:end,:);
lab_unlA = lab_unlA(2:2:end,:);
lab_unlB = lab_unlB(2:2:end,:);

labA = labA'; labB = labB';
unlA = unlA'; unlB = unlB';

cr_dac1 = zeros(3,2);

%% Dom. A

obj = fitcdiscr(labA', lab_labA);

label = predict(obj, unlA');
cr_dac1(1,1) = sum((lab_unlA == label))/numel(label);
label = predict(obj, testA');
cr_dac1(2,1) = sum((lab_testA == label))/numel(label);
label = predict(obj, [unlA'; testA']);
cr_dac1(3,1) = sum(([lab_unlA; lab_testA] == label))/numel(label);

%% Dom. B

obj = fitcdiscr(labB', lab_labB);

label = predict(obj, unlB');
cr_dac1(1,2) = sum((lab_unlB == label))/numel(label);
label = predict(obj, testB');
cr_dac1(2,2) = sum((lab_testB == label))/numel(label);
label = predict(obj, [unlB'; testB']);
cr_dac1(3,2) = sum(([lab_unlB; lab_testB] == label))/numel(label);


cr_dac2 = zeros(3,2);

if (size(labA,1) == size(labB,1))
	obj = fitcdiscr([labA'; labB'], [lab_labA; lab_labB]);

	% Dom. A
	label = predict(obj, unlA');
	cr_dac2(1,1) = sum((lab_unlA == label))/numel(label);
	label = predict(obj, testA');
	cr_dac2(2,1) = sum((lab_testA == label))/numel(label);
	label = predict(obj, [unlA'; testA']);
	cr_dac2(3,1) = sum(([lab_unlA; lab_testA] == label))/numel(label);

	%% Dom. B
	label = predict(obj, unlB');
	cr_dac2(1,2) = sum((lab_unlB == label))/numel(label);
	label = predict(obj, testB');
	cr_dac2(2,2) = sum((lab_testB == label))/numel(label);
	label = predict(obj, [unlB'; testB']);
	cr_dac2(3,2) = sum(([lab_unlB; lab_testB] == label))/numel(label);
else
	% Reduce the longest time series so each domain can be comparable
	if( size(labA,1) < size(labB,1))
		% Linear interpolation to fit length TS
		lab_bisA  = interp1(linspace(0,1,size(labA,1)), labA,  linspace(0,1,size(labB,1)));
		unl_bisA  = interp1(linspace(0,1,size(labA,1)), unlA,  linspace(0,1,size(labB,1)));
		test_bisA = interp1(linspace(0,1,size(labA,1)), testA, linspace(0,1,size(labB,1)));

		obj = fitcdiscr([lab_bisA'; labB'], [lab_labA; lab_labB]);

		% Dom. A
		label = predict(obj, unl_bisA');
		cr_dac2(1,1) = sum((lab_unlA == label))/numel(label);
		label = predict(obj, test_bisA');
		cr_dac2(2,1) = sum((lab_testA == label))/numel(label);
		label = predict(obj, [unl_bisA'; test_bisA']);
		cr_dac2(3,1) = sum(([lab_unlA; lab_testA] == label))/numel(label);

		%% Dom. B
		label = predict(obj, unlB');
		cr_dac2(1,2) = sum((lab_unlB == label))/numel(label);
		label = predict(obj, testB');
		cr_dac2(2,2) = sum((lab_testB == label))/numel(label);
		label = predict(obj, [unlB'; testB']);
		cr_dac2(3,2) = sum(([lab_unlB; lab_testB] == label))/numel(label);
	else
		% Linear interpolation to fit length TS
		lab_bisB  = interp1(linspace(0,1,size(labB,1)), labB,  linspace(0,1,size(labA,1)));
		unl_bisB  = interp1(linspace(0,1,size(labB,1)), unlB,  linspace(0,1,size(labA,1)));
		test_bisB = interp1(linspace(0,1,size(labB,1)), testB, linspace(0,1,size(labA,1)));

		obj = fitcdiscr([labA'; lab_bisB'], [lab_labA; lab_labB]);

		% Dom. A
		label = predict(obj, unlA');
		cr_dac2(1,1) = sum((lab_unlA == label))/numel(label);
		label = predict(obj, testA');
		cr_dac2(2,1) = sum((lab_testA == label))/numel(label);
		label = predict(obj, [unlA'; testA']);
		cr_dac2(3,1) = sum(([lab_unlA; lab_testA] == label))/numel(label);

		%% Dom. B
		label = predict(obj, unl_bisB');
		cr_dac2(1,2) = sum((lab_unlB == label))/numel(label);
		label = predict(obj, test_bisB');
		cr_dac2(2,2) = sum((lab_testB == label))/numel(label);
		label = predict(obj, [unl_bisB'; test_bisB']);
		cr_dac2(3,2) = sum(([lab_unlB; lab_testB] == label))/numel(label);
	end
end

%%

options.ntsperc = N;
options.graph.nn = 5;
options.mu = 1.;
options.fig = 1;
options.d = 10;

%% SSMA

[SS_A_u, SS_A_t, SS_A_ut, SS_B_u, SS_B_t, SS_B_ut] = ssma_xp(labA, lab_labA, unlA, lab_unlA, testA, lab_testA, labB, lab_labB, unlB, lab_unlB, testB, lab_testB, options);

%% KEMA

[KE_A_u, KE_A_t, KE_A_ut, KE_B_u, KE_B_t, KE_B_ut] = kema_xp(labA, lab_labA, unlA, lab_unlA, testA, lab_testA, labB, lab_labB, unlB, lab_unlB, testB, lab_testB, options);

fontname = 'AvantGarde';
fontsize = 20;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
   'DefaultLineLineWidth',2,'DefaultLineMarkerSize',8,'DefaultLineColor',[0 0 0]);
set(findall(gcf,'type','text'), 'fontname', fontname);
set(gcf,'PaperUnits','centimeters');
set(gcf, 'PaperType','A4');
orient landscape;

if(options.fig == 1)
	t = num2cell([1, options.d, 0, 1]);
	[xmin, xmax, ymin, ymax] = deal(t{:});
	
	clf;
	figur = gcf;
	
	v = (1:options.d);
	
	subplot(2,2,1);
	plot(v, SS_A_ut, 'o--'), hold on,
	plot(v, KE_A_ut, 'x-'),  hold on,
	bs_rates = repmat(cr_dac1(3,1), 1,options.d); % unlab+test
	plot(v, bs_rates, ':'),
	bs_rates = repmat(cr_dac2(3,1), 1,options.d);
	plot(v, bs_rates, '-.'),
	ylabel('accuracy'),
	xlabel('dim($\mathcal{F}$)','interpreter','latex'),
	grid off,
	leg_str = {'SSMA', 'KEMA', 'RD-1', 'RD-2'};
	leg = columnlegend(2, leg_str);
	set(leg, 'position', [0.05,0.8,0.5,0.1] );
	axis([xmin xmax ymin ymax]);
	
	subplot(2,2,2);
	h(1) = plot(v, SS_B_ut, 'o--'); hold on,
	h(2) = plot(v, KE_B_ut, 'x-'); hold on,
	bs_rates = repmat(cr_dac1(3,2), 1,options.d);
	h(3) = plot(v, bs_rates, ':');
	bs_rates = repmat(cr_dac2(3,2), 1,options.d);
	h(4) = plot(v, bs_rates, '-.');
	ylabel('accuracy'),
	xlabel('dim($\mathcal{F}$)','interpreter','latex'),
	grid off,
	leg_str = {'SSMA', 'KEMA', 'RD-1', 'RD-2'};
	leg = columnlegend(2, leg_str);
	set(leg, 'position', [0.5,0.8,0.5,0.1] );
	axis([xmin xmax ymin ymax]);
	
	saveas(figur, 'classif_rates', 'pdf');
	clear figur;
	
	disp('A')
	disp([cr_dac1(3,1), cr_dac2(3,1), SS_A_ut(5), KE_A_ut(5)])
	disp('B')
	disp([cr_dac1(3,2), cr_dac2(3,2), SS_B_ut(5), KE_B_ut(5)])

	%disp('Press enter to continue'); pause
end

%clear all
