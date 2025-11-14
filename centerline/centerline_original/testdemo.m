clear all
close all
clc
load image_all.mat
figure,XimShow(image_all)
Point1=[245,306,247];
Point2=[267,314,420];
[ThreMax1,ThreMin1]=AutomaticThresholding1(image_all,Point1,1);
[ThreMax2,ThreMin2]=AutomaticThresholding1(image_all,Point2,1);
ThreMax=max(ThreMax1,ThreMax2);
ThreMin=min(ThreMin1,ThreMin2);

figure,XimShow((image_all<=ThreMax)&(image_all>=ThreMin));

