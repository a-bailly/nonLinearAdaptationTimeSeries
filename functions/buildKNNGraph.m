%-------------------------------------------------------------------
% Build a knn graph
%
% dist = buildKNNGraph(data,k,type);
%
% Input:
%    - data :      the input data matrix
%    - k    :      number of neighbors
%    - type :      type of graph (0: mutual, 1: at least one direction)
%              
% Output:
%    - G    :      the graph
%
% Devis Tuia
% 
% devis.tuia@unil.ch
%  
% Inspred by the work of Fr�d�ric Ratle
%-------------------------------------------------------------------

function dist = buildKNNGraph(data,k,type);

dist  = L2_distance(data',data');
[A, IX] = sort(dist,1);

%Gives 1 to the k nearest neighbors and 0 to the others
for j = 1:size(dist,2)
    dist(IX(2:k+1,   j), j) = 1;
    dist(IX(k+2:end, j), j) = 0;
end

dist = sparse(dist);

if type == 0
    dist = min(dist,dist');
else
    dist = max(dist,dist');
end

