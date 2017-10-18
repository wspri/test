%%
% clear all;close all;clc;
% load train.txt;

inputSeries = tonndata(inputn,true,false);%��ѵ������ת��Ϊ��������õı�׼�ṹ
targetSeries = tonndata(outputn,true,false);

vad_input= tonndata(inputn_test,true,false);%����������ת��Ϊ��������õı�׼�ṹ
vad_output= tonndata(outputn_test,true,false);

%%
%����NARX���磬Ĭ��tan-sigmoidΪ����ת�ƺ�����linearΪ�����ת�ƺ�������������
%�������룬һ�����ⲿ���룬һ�������������ѵ����ɺ������ӽ����Ͽ�����
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