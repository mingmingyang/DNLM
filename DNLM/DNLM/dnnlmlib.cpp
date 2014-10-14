//
//  dnnlmlib.cpp
//  DNLM
//
//  Created by ymm on 14-10-11.
//  Copyright (c) 2014年 ymm. All rights reserved.
//

#include "dnnlmlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static union{
    double d;
    struct{
        int j,i;
    } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i=EXP_A*(y)+(1072693248-EXP_C),d2i.d)

///include blas
#ifdef USE_BLAS
extern "C"{
#include <cblas.h>
}
#endif
//

real CDnnLM::random(real min,real max)
{
    return rand()/(real)RAND_MAX*(max-min)+min;
}

void CDnnLM::setTrainFile(char *str)
{
    strcpy(train_file,str);
}

void CDnnLM::setValidFile(char *str)
{
    strcpy(valid_file,str);
}

void CDnnLM::setTestFile(char *str)
{
    strcpy(test_file,str);
}

void CDnnLM::setDnnLMFile(char *str)
{
    strcpy(dnnlm_file,str);
}

void CDnnLM::readWord(char *word, FILE *fin)
{
    int a=0,ch;
    while(!feof(fin))
    {
        ch=fgetc(fin);
        if (ch==13) continue;
        
        if ((ch==' ')||(ch=='\t')||(ch=='\n'))
        {
            if (a>0){
                if (ch=='\n') ungetc(ch,fin);
                break;
            }
            
            if (ch=='\n')
            {
                strcpy(word, (char *)"</s>");
                return;
            }
            else continue;
        }
        
            word[a]=ch;
            a++;
            if(a>=MAX_STRING)
            {
                a--;
            }
    }
    word[a]=0;
}

int CDnnLM::getWordHash(char *word)
{
    unsigned int hash,a;
    hash=0;
    for(a=0;a<strlen(word);a++)
    {
     hash=hash*237+word[a];
    }
    hash=hash%vocab_hash_size;
    return hash;
}

int CDnnLM::searchVocab(char *word)
{
    int a;
    unsigned int hash;
    hash=getWordHash(word);
    if(vocab_hash[hash]==-1)
    {
        return -1;
    }
    if(!strcmp(word,vocab[vocab_hash[hash]].word))
    {
        return vocab_hash[hash];
    }
    for(a=0;a<vocab_size;a++)
    {
        if(!strcmp(word,vocab[a].word))
        {
            vocab_hash[hash]=a;
            return a;
        }
    }
    return -1;
}

int CDnnLM::readWordIndex(FILE *fin)
{
    char word[MAX_STRING];
    
    readWord(word,fin);
    
    if(feof(fin))
    {
        return -1;
    }
    return searchVocab(word);
}

int CDnnLM::addWordToVocab(char *word)
{
    unsigned int hash;
    
    strcpy(vocab[vocab_size].word,word);
    vocab[vocab_size].cn=0;
    vocab_size++;
    
    if (vocab_size+2>=vocab_max_size)
    {
        vocab_max_size+=100;
        vocab=(struct vocab_word *)realloc(vocab,vocab_max_size*sizeof(struct vocab_word));
        
    }
    hash=getWordHash(word);
    vocab_hash[hash]=vocab_size-1;
    return vocab_size-1;
}

void CDnnLM::sortVocab()
{
    int a,b,max;
    vocab_word swap;
    for (a=1;a<vocab_size;a++)
    {
        max=a;
        for(b=a+1;b<vocab_size;b++)
        {
            if(vocab[max].cn<vocab[b].cn)
            {
                max=b;
            }
        }
        swap=vocab[max];
        vocab[max]=vocab[a];
        vocab[a]=swap;
    }
}

void CDnnLM::learnVocabFromTrainFile()
{
    char word[MAX_STRING];
    FILE *fin;
    int a,i,train_wcn;
    for(a=0;a<vocab_hash_size;a++)
    {
        vocab_hash[a]=-1;
    }
    fin=fopen(train_file,"rb");
    
    vocab_size=0;
    
    addWordToVocab((char *)"</s>");
    
    train_wcn=0;
    while(1)
    {
        readWord(word,fin);
        if (feof(fin))
        {
            break;
        }
        train_wcn++;
        
        i=searchVocab(word);
        if(i==-1)
        {
            a=addWordToVocab(word);
            vocab[a].cn=1;
            
        }
        else
        {
            vocab[i].cn++;
        }
    }
    
    sortVocab();
    
    if(debug_mode>0)
    {
        printf("Vocab size: %d\n",vocab_size);
        printf("Words in train file: %d\n",train_wcn);
    }
    train_words=train_wcn;
    fclose(fin);
}
