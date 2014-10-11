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
        
    }
    
    
};
#include <stdio.h>

#endif /* defined(__DNLM__dnnlmlib__) */
