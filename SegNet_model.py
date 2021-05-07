import torch
import torch.nn as nn
import torchvision.models as models


F = nn.functional

class SegNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SegNet,self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.vgg16 = models.vgg16(pretrained= True)

        # Encoder layers 
        self.encoder_1 = nn.Sequential(
                                *[
                                    nn.Conv2d(in_channels= self.in_channels,out_channels= 64,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),        
                                    nn.Conv2d(in_channels= 64,out_channels= 64,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU()        
                                ]
                             )
        self.encoder_2 = nn.Sequential(
                                *[
                                    nn.Conv2d(in_channels= 64,out_channels= 128,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),        
                                    nn.Conv2d(in_channels= 128,out_channels= 128,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()        
                                ]
                             )
        self.encoder_3 = nn.Sequential(
                                *[
                                    nn.Conv2d(in_channels= 128,out_channels= 256,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),        
                                    nn.Conv2d(in_channels= 256,out_channels= 256,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels= 256,out_channels= 256,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU()        
                                ]
                             )   
        self.encoder_4 = nn.Sequential(
                                *[
                                    nn.Conv2d(in_channels= 256,out_channels= 512,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),        
                                    nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU()        
                                ]
                             )
        self.encoder_5 = nn.Sequential(
                                *[
                                    nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),        
                                    nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size= 3,padding = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU()        
                                ]
                             )                        
        self.init_vgg__weights() # load the weight of pretrained model into our encoder network

        self.decoder_1 = nn.Sequential(
                                *[
                                    nn.ConvTranspose2d(in_channels=512, out_channels= 512,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=512, out_channels= 512,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=512, out_channels= 512,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU()
                                ]
        ) 
        self.decoder_2 = nn.Sequential(
                                *[
                                    nn.ConvTranspose2d(in_channels=512, out_channels= 512,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=512, out_channels= 512,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=512, out_channels= 256,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU()
                                ]
        ) 
        self.decoder_3 = nn.Sequential(
                                *[
                                    nn.ConvTranspose2d(in_channels=256, out_channels= 256,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=256, out_channels= 256,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=256, out_channels= 128,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()
                                ]
        ) 
        self.decoder_4 = nn.Sequential(
                                *[
                                    nn.ConvTranspose2d(in_channels=128, out_channels= 128,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=128, out_channels= 64,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU()
                                ]
        ) 
        self.decoder_5 = nn.Sequential(
                                *[
                                    nn.ConvTranspose2d(in_channels=64, out_channels= 64,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=64, out_channels= self.out_channels,kernel_size=3,padding=1)
                                ]
        ) 


    def forward(self,input_img):
        # Forwrad pass for encoder 
        dim0 = input_img.size()
        out = self.encoder_1(input_img)
        out, indx1 = nn.MaxPool2d(out,kernel_size = 2, stride= 2, return_indices= True)

        dim1 = out.size()
        out = self.encoder_2(out)
        out,indx2 = nn.MaxPool2d(out,kernel_size = 2, stride= 2, return_indices= True)

        dim2 = out.size()
        out = self.encoder_3(out)
        out,indx3 = nn.MaxPool2d(out,kernel_size = 2, stride= 2, return_indices= True)

        dim3 = out.size()
        out = self.encoder_4(out)
        out,indx4 = nn.MaxPool2d(out,kernel_size = 2 ,stride= 2, return_indices= True)

        dim4 = out.size()
        out = self.encoder_5(out)
        out,indx5 = nn.MaxPool2d(out,kernel_size = 2 ,stride= 2, return_indices= True)

        dim5 = out.size()
        # decoder 
        # unpooling the correspondig layer 
        out = nn.MaxUnpool2d(out,indx5,kernel_size = 2,stride= 2, output_size= dim4)
        out  = self.decoder_1(out)

        out = nn.MaxUnpool2d(out,indx4,kernel_size = 2,stride= 2, output_size= dim3)
        out  = self.decoder_2(out)

        out = nn.MaxUnpool2d(out,indx3,kernel_size = 2,stride= 2, output_size= dim2)
        out  = self.decoder_3(out)

        out = nn.MaxUnpool2d(out,indx2,kernel_size = 2,stride= 2, output_size= dim1)
        out  = self.decoder_4(out)

        out = nn.MaxUnpool2d(out,indx1,kernel_size = 2,stride= 2, output_size= dim0)
        out  = self.decoder_5(out)

        softmax_out = nn.Softmax(out,dim = 1)

        return out, softmax_out

    def init_vgg__weights(self):

       # print(self.encoder_1[3].weight.size(), self.vgg16.features[0].weight.size())
       #Encoder 1
       assert self.encoder_1[0].weight.size() == self.vgg16.features[0].weight.size()
       self.encoder_1[0].weight.data = self.vgg16.features[0].weight.data
       self.encoder_1[0].bias.data = self.vgg16.features[0].bias.data    
       assert self.encoder_1[3].weight.size() == self.vgg16.features[2].weight.size()
       self.encoder_1[3].weight.data = self.vgg16.features[2].weight.data
       self.encoder_1[3].bias.data = self.vgg16.features[2].bias.data

       #Encoder 2
       assert self.encoder_2[0].weight.size() == self.vgg16.features[5].weight.size()
       self.encoder_2[0].weight.data = self.vgg16.features[5].weight.data
       self.encoder_2[0].bias.data = self.vgg16.features[5].bias.data    
       assert self.encoder_2[3].weight.size() == self.vgg16.features[7].weight.size()
       self.encoder_2[3].weight.data = self.vgg16.features[7].weight.data
       self.encoder_2[3].bias.data = self.vgg16.features[7].bias.data

       #Encoder 3
       assert self.encoder_3[0].weight.size() == self.vgg16.features[10].weight.size()
       self.encoder_3[0].weight.data = self.vgg16.features[10].weight.data
       self.encoder_3[0].bias.data = self.vgg16.features[10].bias.data    
       assert self.encoder_3[3].weight.size() == self.vgg16.features[12].weight.size()
       self.encoder_3[3].weight.data = self.vgg16.features[12].weight.data
       self.encoder_3[3].bias.data = self.vgg16.features[12].bias.data
       assert self.encoder_3[6].weight.size() == self.vgg16.features[14].weight.size()
       self.encoder_3[6].weight.data = self.vgg16.features[14].weight.data
       self.encoder_3[6].bias.data = self.vgg16.features[14].bias.data

       #Encoder 4
       assert self.encoder_4[0].weight.size() == self.vgg16.features[17].weight.size()
       self.encoder_4[0].weight.data = self.vgg16.features[17].weight.data
       self.encoder_4[0].bias.data = self.vgg16.features[17].bias.data    
       assert self.encoder_4[3].weight.size() == self.vgg16.features[19].weight.size()
       self.encoder_4[3].weight.data = self.vgg16.features[19].weight.data
       self.encoder_4[3].bias.data = self.vgg16.features[19].bias.data    
       assert self.encoder_4[6].weight.size() == self.vgg16.features[21].weight.size()
       self.encoder_4[6].weight.data = self.vgg16.features[21].weight.data
       self.encoder_4[6].bias.data = self.vgg16.features[21].bias.data

       #Encoder 5
       assert self.encoder_5[0].weight.size() == self.vgg16.features[24].weight.size()
       self.encoder_5[0].weight.data = self.vgg16.features[24].weight.data
       self.encoder_5[0].bias.data = self.vgg16.features[24].bias.data    
       assert self.encoder_5[3].weight.size() == self.vgg16.features[26].weight.size()
       self.encoder_5[3].weight.data = self.vgg16.features[26].weight.data
       self.encoder_5[3].bias.data = self.vgg16.features[26].bias.data    
       assert self.encoder_5[6].weight.size() == self.vgg16.features[28].weight.size()
       self.encoder_5[6].weight.data = self.vgg16.features[28].weight.data
       self.encoder_5[6].bias.data = self.vgg16.features[28].bias.data    


if __name__ == "__main__":

    model = SegNet(3,21)

    model.init_vgg__weights()
    print("DONE")




