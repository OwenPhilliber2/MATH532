cities = {'Denver','Fort Collins','Breckenridge','Copper Mtn','Steamboat', ...
          'Aspen','Winter Park','Walden','Snowy Range'};

D = [ ...
    0   65   84   87  156  199   66  142  161;  % Denver
   65    0  150  138  185  236  126   99  120;  % Fort Collins
   84  150    0   16  103  139   66  113  235;  % Breckenridge
   87  138   16    0  100   82   64  111  181;  % Copper Mtn
  156  185  103  100    0  155  100   59  123;  % Steamboat
  199  236  139   82  155    0  144  198  280;  % Aspen
   66  126   66   64  100  144    0   77  140;  % Winter Park
  142   99  113  111   59  198   77    0   71;  % Walden
  161  120  235  181  123  280  140   71    0]; % Snowy Range

% Optional: put it in a labeled table
T = array2table(D, 'RowNames', cities, 'VariableNames', matlab.lang.makeValidName(cities));
