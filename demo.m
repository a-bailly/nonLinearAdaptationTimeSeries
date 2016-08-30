%% @author: Adeline Bailly adeline.bailly@univ-rennes2.fr
%% @date: 2016
%% Based on KEMA from: https://github.com/dtuia/KEMA

%% clear
clear; clc;
close all;

%% spec
path = '../time_series_data/';
name_domA = 'belmanip/fapar.txt';
name_domB = 'belmanip/lai.txt';
NN = 5;

%'gee/modis_eu_ndvi_8day_2011.txt'; % 

%'gee/modis_eu_ndvi_8day_2003.txt'; %
%'gee/modis_na_ndvi_8day_2011.txt'; %
%'gee/modis_eu_ndvi_16day_2011.txt'; %
%'gee/landsat_eu_ndvi_8day_2011.txt'; %

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

[labels, ~, ic] = unique(labelsA);
disp('number of class :'),
disp(numel(labels));

disp('L1: labels ; L2: Domain A Occurences ; L3: Domain B Occurences');
ha = histc(ic, unique(ic));
[labels, ~, ic] = unique(labelsB);
hb = histc(ic, unique(ic));
disp([labels'; ha'; hb']);

disp('-----------------------------------')
disp('-- Classif of A and B using both --')
disp('-----------------------------------')
ma_predict(domA, labelsA, domB, labelsB, NN);
disp('-------------------------')
disp('--Classif of A using B --')
disp('-------------------------')
ma_predictAwB(domA, labelsA, domB, labelsB, NN);
