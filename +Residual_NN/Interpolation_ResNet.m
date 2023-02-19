lgraph = layerGraph();

tempLayers = [
    imageInputLayer(Input_Layer_Size,"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],8,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],8,"Name","conv_3","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],8,"Name","conv_4","Padding","same")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],8,"Name","conv_5","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],8,"Name","conv_6","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],8,"Name","conv_7","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],8,"Name","conv_8","Padding","same")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],8,"Name","conv_9","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([3 3],8,"Name","conv_10","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(6,"Name","addition_5")
    resize2dLayer("Name","resize-output-size","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","OutputSize",[72 14])
    convolution2dLayer([36 7],2,"Name","conv","Padding","same")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"conv_1","conv_2");
lgraph = connectLayers(lgraph,"conv_1","addition_1/in1");
lgraph = connectLayers(lgraph,"conv_1","addition_5/in2");
lgraph = connectLayers(lgraph,"conv_3","addition_1/in2");
lgraph = connectLayers(lgraph,"addition_1","conv_4");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in1");
lgraph = connectLayers(lgraph,"addition_1","addition_5/in3");
lgraph = connectLayers(lgraph,"conv_5","addition_2/in2");
lgraph = connectLayers(lgraph,"addition_2","conv_6");
lgraph = connectLayers(lgraph,"addition_2","addition_3/in1");
lgraph = connectLayers(lgraph,"addition_2","addition_5/in4");
lgraph = connectLayers(lgraph,"conv_7","addition_3/in2");
lgraph = connectLayers(lgraph,"addition_3","conv_8");
lgraph = connectLayers(lgraph,"addition_3","addition_4/in1");
lgraph = connectLayers(lgraph,"addition_3","addition_5/in5");
lgraph = connectLayers(lgraph,"conv_9","addition_4/in2");
lgraph = connectLayers(lgraph,"addition_4","conv_10");
lgraph = connectLayers(lgraph,"addition_4","addition_5/in6");
lgraph = connectLayers(lgraph,"conv_10","addition_5/in1");