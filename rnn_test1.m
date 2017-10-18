%%
% clear all;close all;clc;
% load train.txt;

inputSeries = tonndata(inputn,true,false);%将训练数据转换为神经网络可用的标准结构
targetSeries = tonndata(outputn,true,false);

vad_input= tonndata(inputn_test,true,false);%将测试数据转换为神经网络可用的标准结构
vad_output= tonndata(outputn_test,true,false);

%%
%创建NARX网络，默认tan-sigmoid为隐层转移函数，linear为输出层转移函数。该网络有
%两个输入，一个是外部输入，一个是输出反馈（训练完成后反馈连接将被断开）。
inputDelays=1:2;
feedbackDelays=1:2;
hiddenLayerSize=10;
net=narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
%%
[inputs,inputStates,layerStates,targets] = ...
    preparets(net,inputSeries,{},targetSeries);
%%
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;

%%
[net,tr] = train(net,inputs,targets,inputStates,layerStates);
%%
nets=removedelay(net);

[inputs,inputStates,layerStates,targets] = ...
    preparets(nets,vad_input,{},vad_output);
%%
outputs = nets(inputs,inputStates,layerStates);
%%
plot(cell2mat(outputs));
hold on
plot(cell2mat(vad_output));
