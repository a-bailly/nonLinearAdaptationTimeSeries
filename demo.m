%% @author: Adeline Bailly adeline.bailly@univ-rennes2.fr
%% @date: 2016
%% Based on KEMA from: https://github.com/dtuia/KEMA

%% clear
clear; clc;
close all;

%% spec
path = '../time_series_data/';

	%% GEE-TSDA
name_domA = 'gee_tsda/modis_sa_ndvi_8day_2011.txt'; % South America
%name_domA = 'gee_tsda/modis_na_ndvi_8day_2011.txt'; % North America
%name_domA = 'gee_tsda/modis_eu_ndvi_8day_2003.txt'; % 2003
%name_domA = 'gee_tsda/landsat_eu_ndvi_8day_2011.txt'; % Lansat
%name_domA = 'gee_tsda/modis_eu_lai_4day_2011.txt'; % Lai 4day
name_domB = 'gee_tsda/modis_eu_ndvi_8day_2011.txt'; % 

	%% BELMANIP
%name_domA = 'belmanip/fapar.txt'; %
%name_domB = 'belmanip/fvc.txt'; %
NN = 5;

addpath(genpath('functions'));

%% read data

% Domain A
s = strcat(path, name_domA);
domA = load(s);
labelsA = domA(:,1);
domA = domA(:,2:end);

% Domain B
s = strcat(path, name_domB);
domB = load(s);
labelsB = domB(:,1);
domB = domB(:,2:end);

labels = unique(labelsA);
disp('number of class :'),
disp(numel(labels));

disp('L1: labels ; L2: Domain A Occurences ; L3: Domain B Occurences');
	%% Check if each domains contain same classes
labels = unique(labelsA);
labelsBb = unique(labelsB);
if(length(labels) ~= length(labelsBb))
	new_labels = intersect(labels, labelsBb);
	for ii=1:length(labels)
		if(~ismember(labels(ii), new_labels))
			k = find(labelsA~=labels(ii));
			domA = domA(k,:);
			labelsA = labelsA(k);
		end
	end
	for ii=1:length(labelsBb)
		if(~ismember(labelsBb(ii), new_labels))
			k = find(labelsB~=labelsBb(ii));
			domB = domB(k,:);
			labelsB = labelsB(k);
		end
	end
end

[labels, ~, ic] = unique(labelsA);
ha = histc(ic, unique(ic));
[labelsBb, ~, ic] = unique(labelsB);
hb = histc(ic, unique(ic));

disp([labels'; ha'; hb']);

disp('-----------------------------------')
run_xp(domA, labelsA, domB, labelsB, NN);
disp('-----------------------------------')

fig_mean_profil = 0;
if(fig_mean_profil == 1)
	ff = figure(2);
	x=1:size(domA,2);
	colormap(jet(numel(labels)));
	linestyle = {':', '-', '--', '-.'};
	for ii=1:numel(labels)
		k = find(labelsA==labels(ii));
		plot(x, mean(domA(k,:))', 'LineStyle', linestyle{mod(ii,numel(linestyle))+1}), hold on;
	end
	leg_str = {'Evergreen', 'Deciduous', 'Shrublands', 'Savannas', 'Grasslands', 'Croplands'}; %
	leg = columnlegend(3, leg_str);
	%set(leg, 'position', [0.1,0.05,0.9,0.1] ); % bottom
	set(leg, 'position', [0.1,0.7,0.9,0.1] ); % top
	hold off;
	axis([0 size(domA,2) 0 1]); % put 70 for LAI
	saveas(ff, '_mean_ts', 'pdf');
end
