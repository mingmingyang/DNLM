//
//  dnnlmlib.h
//  DNLM
//
//  Created by ymm on 14-10-11.
//  Copyright (c) 2014年 ymm. All rights reserved.
//

#ifndef __DNLM__dnnlmlib__
#define __DNLM__dnnlmlib__
#define MAX_STRING 100
#include <stdio.h>
#include <stdlib.h>
typedef double real;// doubles for NN weights
//typedef double direct_t;

struct neuron{
    real ac;//actual value stored in neuron
    real er;//error value in neuron, used by learning algorithm
};

struct synapse{
    real weight; //weight of synapse
};

struct vocab_word{
    int cn;
    char word[MAX_STRING];
    real prob;
    int class_index;
};

const int MAX_NGRAM_ORDER=20;

class CDnnLM{
protected:
    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char dnnlm_file[MAX_STRING];
    char lmprob_file[MAX_STRING];
    
    int debug_mode;
    int version;
    int filetype;
    int use_lmprob;
    real lambda;
    real alpha;
    int iter;
    int vocab_max_size;
    int vocab_size;
    int train_words;
    int train_cur_pos;
    int counter;
    int one_iter;
    real beta;
    int class_size;
    int **class_words;
    int *class_cn;
    int *class_max_cn;
    int old_classes;
    struct vocab_word *vocab;
    void sortVocab();
    int *vocab_hash;
    int vocab_hash_size;
    int layer0_size;
    int layer1_size;
    int layer2_size;
    long long direct_size;
    float min_improvement;
    struct neuron *neu0;// neurons in input layer
    struct neuron *neu1;// neurons in hidden layer
    struct neuron *neu2;//neurons in output layer
    
    struct synapse *syn0;//weights between input and hidden layer
    struct synapse *syn1;//weights between hidden and output layer

public:
    int alpha_set,train_file_set;
    CDnnLM() //constructor initializes variables
    {
        version=0;
        use_lmprob=0;
        train_file[0]=0;
        valid_file[0]=0;
        test_file[0]=0;
        dnnlm_file[0]=0;
        train_file_set=0;
        alpha=0.1;
        beta=0.0000001;
        iter=0;
        min_improvement=1.003;
        train_words=0;
        train_cur_pos=0;
        vocab_max_size=100;
        vocab_size=0;
        vocab=(struct vocab_word *)calloc(vocab_max_size,sizeof(struct vocab_word));
        layer1_size=30;
        
        neu0=NULL;
        neu1=NULL;
        neu2=NULL;
        
        syn0=NULL;
        syn1=NULL;
        class_size=100;
        one_iter=0;
        debug_mode=1;
        
        vocab_hash_size=100000000;
        vocab_hash=(int *)calloc(vocab_hash_size,sizeof(int));
        
    }
    ~CDnnLM()
    {
        int i;
        if (neu0!=NULL) {
            free(neu0);
            free(neu1);
            free(neu2);
            
            free(syn0);
            free(syn1);
            
            for (i=0;i<class_size;i++)
            {
                free(class_words[i]);
            }
                 free(class_max_cn);
                 free(class_cn);
                 free(class_words);
                
                 free(vocab);
                 free(vocab_hash);
        }
    }
    
    real random(real min,real max);
    
    void setTrainFile(char *str);
    void setValidFile(char *str);
    void setTestFile(char *str);
    void setDnnLMFile(char *str);
    void setFileType(int newt){filetype=newt;}
    void setClassSize(int newSize){class_size=newSize;}
    void setLambda(real newLambda){lambda=newLambda;}
    
    void setLearningRate(real newAlpha){alpha=newAlpha;}
    void setRegularization(real newBeta){beta=newBeta;}
    void setMinImprovement(real newMinImprovement){min_improvement=newMinImprovement;}
    void setHiddenLayerSize(int newsize){layer1_size=newsize;}
    void setDebugMode(int newDebug){debug_mode=newDebug;}
    void setOneIter(int newOneIter){one_iter=newOneIter;}
    
    int getWordHash(char *word);
    void readWord(char *word, FILE *fin);
    int searchVocab(char *word);
    int readWordIndex(FILE *fin);
    int addWordToVocab(char *word);
    void learnVocabFromTrainFile();
    
    void saveWeights();
    void restoreWeights();
    
    void initNet();
    void saveNet();
    void goTodelimiter(int delim,FILE *fi);
    void restoreNet();
    void netFlush();
    void netReset();
    
    void computeNet(int last_word,int word);
    void learnNet(int last_word,int word);
    void trainNet();
    void testNet();
    void matrixXvector(struct neuron *dest,struct neuron *srcvec,struct synapse *srcmatrix,int matrix_width,int from,int to,int from2,int to2,int type);
};


#endif /* defined(__DNLM__dnnlmlib__) */
