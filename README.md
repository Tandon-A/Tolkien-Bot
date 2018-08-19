# Tolkien Bot 

Tensorflow implementation of a character level language model to generate texts. 


## Prerequisites

* Python 3.3+
* Tensorflow 1.6+
* Training Data

## Model Architecture 

The model used in this project is a 4 layer LSTM model with 512 neurons in each layer. 

## Usage

To train the model:
```
> python train.py --file_path train_file.txt --model_dir sbot_model/ --meta_dir meta_dir/
```
* file_path: Path of the training text file
* model_dir: Path of the directory where model files would be saved. (Default: "model_dir/")
* meta_dir: Path of the directory where training file metadata would be stored. (Default: "meta_dir/")


To test the model:
```
> python test.py --meta_dir meta_dir/ --model_dir sbot_model/
```
* meta_dir: Path of the directory where training file metadata is stored. 
* model_dir: Path of the directory where model files are stored. 


## Results 

Trained LSTM Model on Lord of the Rings. 

### Model Output 1 

>A  and of the Elves  see  
     And the sound of the  horr  
     And the  the sale the  hound of the sall 
     And the  soon of the  old of the  Calrow  
     And the  soon of the  ord  was  searing  and  salled  and  seemed  to  the  sea  
     and the sear of the  freen 
     and the dark side of the sall 
     And see the sound of the wold 
     And tare and seen a     The  hair of the wold 
     And the  word of the  hill  
     And the  soot of the  Countains 
     And the  sang was seared and salked aand and sell 
     And sat the sound of the wold 
     And talled and salk and salked 
     And tore and slowled and set in a slade, 
     And the  song af the sall was slawe  
     and the dark of the dark 
     and see  and seen 
     And the soor of the sall 
     And the land of the  word  
     And tared and sat in the same 
     And the sigh was  all and sended 
     And tall and seep and salled 
     and saiked and saled 
     And net the sound of the door and seen 



### Model Output 2 


>shadows in the matthr of the root. It was a  
splender  of  the  trees,  only  that  they were  still  and  their  campany speak  and 
went along a standing shadow of the  thees like a  shadow of the 
dimrer  in the trees. 
     \`Ahen  as  they  were  and  said,  and  the  south  of  the  stream  of  the 
Lord of  Lady  full  sun in  the  stream  and  fear  the  halls  of  the  trees 
and  the  world  of  the  Lady  and  the  water  sat  and  time  of  course,  and  they 
saw a  light  of  the  great  clat mountains  in  the  tall  stars  of  the  days  that  they said 
the  wizard  fingered  and was  still  a  shadow  behind  them.  The  wizard  was  long again,  and  leaping  the 
rrand  to  the  days  of  the  Lady  Mountains.  The  end of the  dear  was  still  silver of 
the  same  seemed  to be  his  silence  and  heard  and  the dwarves  was said  for a white and great 
silver  had  crossed  the  hend  of  the  Great  Riders  and  the  trees  were  green


### Model Ouput 3

>rodo. \`I fear our foge, but I am 
  said  Bilbo. 'It is not that we will be to see almost of for a distrebate  to fond of the sound of sourh. It's not 
forgotten  it  without  us and cay to  get to can no 
know  that even if this is  a long party return. I  have  seen their heart in the land,' said Bilbo. 'The  doors was 
not snown  of  the elf-town. We must  seem it sorth and I have gate the evil  to him. I do not ind whisper: I have his  dogs  for  my  world. I  have been known to tild you  of  the  root and  he was   glimper  and  darkness, the side 
of Gerifn will seal it, who was a  
inger it and be affer  the  buttles and the  words of the Sun rather  are  that  is not  say out  lands,  and  there is  head 
trees.  I think I took  to talk and was Gandalf and he had to go of at least. ' 
     \`Well, Grodn  and  Pippin  and  Kili in the East Rnole as deeped from the Elves and 
the bridge of Lurien was a spear of green and sat the trail of half. He saw a  
strange  to  be  seen  his


### Experiment

Trained the model on Queen songs. 

### Model Output Queen 

```
'm live mat, I'll leav your len rt wes

Whme you say to live (oon't lo e

And I'm young do is do it live is born litely love

And I love in all you'me all

Woist love is in you, live se leavino head

Whan it all you gonna los to han love

I wont it all I want it all I wan' toose no fall you

I can t beat th the sige to everything wo lo lo so e nee doh

Let me san , hoin alone and teep a d roeds, IAll naiding times and teal

I fan't live, it slw you wead tt all on time

Iell the eme of mash

Yo yo
```

## Acknowledgement

* [Andrej Karpathy's blog on CharRNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

* [Parsing Blog](https://www.digitalocean.com/community/tutorials/how-to-scrape-web-pages-with-beautiful-soup-and-python-3) - Used to parse Queen songs. 


## License

This project is licensed under the MIT License - see the [LICENSE]() file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
