from flask import Flask, redirect, url_for, request,render_template,jsonify
import json,os,sys
import numpy as np
import pandas as pd
import random

sys.path.append(r'./BotProject')
from BotLink import bot_follow

bf=bot_follow()
app = Flask('to_follow')
#app.config['JSON_AS_ASCII'] = False
app.secret_key='\x03\x8do\xea\xbfuQ\xbbW\xdd\xace\x85\xa6\x8bV\x1b\x1fg\xf4\xe3Ke\x0f'

def to_follow_main(json_data):
    bot_name = json_data['bot']
    name_list=[x.strip() for x in json_data['name_list']]#list of str
    score={}
    embs=bf.get_embs('embs.npy')
    user_map=bf.name2node
    # write input name list 
    bot_node,node_list=save_input(user_map,name_list,bot_name,file = 'nameid.csv')
    # get result
    result={}
    if bf.check_name(bot_node,embs):
        score,recomm=bf.get_score_dict(embs,user_map,bot_node,name_list)
        df_score=pd.DataFrame([score,recomm],index=['score','label'])
        df_score=df_score.T
        df_score=df_score.sort_values(by = 'score',ascending = False)
        df_score.to_csv('out.csv')
        result=df_score.to_dict()
        return bf.to_jsonstr(result)
    else:
        result["Error"]="Bot without embedding, pls choose another bot's name"
        result['score']={}
        return bf.to_jsonstr(result)

def to_follow_train(json_data):
    bot_name = json_data['bot']
    name_list=[x.strip() for x in json_data['name_list']]#list of str
    score={}
    embs=bf.embs
    user_map=bf.name2node
    # write input name list 
    bot_node,node_list=save_input(user_map,name_list,bot_name,file = 'nameid.csv')
    # get result
    result={}
    if bf.check_name(bot_node,embs):
        score,_=bf.get_score_dict(embs,user_map,bot_node,name_list)
        G=bf.G
        # label
        true_label={}
        for i,name in enumerate(name_list):
            node=node_list[i]
            true_label[name]=bf.get_label(G,bot_node,node)
            print(name,node)
        # creat output dataframe 
        df_score=pd.DataFrame([score,true_label],index=['score','label'])
        df_score=df_score.T
        df_score=df_score.sort_values(by = 'score',ascending = False)
        df_score.to_csv('out.csv')# 结果保存
        result=df_score.to_dict()
        return bf.to_jsonstr(result)
    else:
        result["Error"]="Bot without embedding, pls choose another bot's name"
        result['score']={}
        return bf.to_jsonstr(result)

def save_input(user_map,name_list,bot_name,file = 'nameid.csv'):
    node_list=[user_map[x] for x in name_list]
    bot_node=user_map[bot_name]
    dictx={'name':[bot_name]+name_list,'id':[bot_node]+node_list}
    index=['bot']+['target_'+str(i) for i in range(len(name_list))]
    df=pd.DataFrame(dictx,index=index)
    df.to_csv(file)
    return (bot_node,node_list)

def get_names(json_data):
    num= int(json_data['num'])
    nodes=list(bf.Nodes)
    bot=random.choice(nodes)
    botname=bf.node2name[bot]
    nodes.remove(bot)
    nodelist=random.sample(nodes,num-1)
    if list(bf.G.successors(bot)):
        nodelist.append(random.choice(list(bf.G.successors(bot))))
    else:
        nodelist.append(random.sample(nodes,1))
    name_list=[bf.node2name[i] for i in nodelist]
    result={'bot':botname,'name_list':name_list}
    print(result)
    return bf.to_jsonstr(result)

@app.route('/')
def index():
    msg={}
    msg['test_link']='success'
    return jsonify(msg)

@app.route('/to_follow/beta',methods = ['POST','GET'])
def to_follow():
    print('----- Server beta Start-----')
    print(request.headers)
    print(request.form)
    print(type(request.json))
    print(request.json)
    return to_follow_main(request.json)

@app.route('/to_follow/v2',methods = ['POST','GET'])
def to_follow_d():
    print('----- Server v2 Start-----')
    print(request.headers)
    print(request.form)
    print(type(request.json))
    print(request.json)
    return to_follow_train(request.json)

@app.route('/to_follow/get_random',methods = ['POST','GET'])
def get_random_name():
    print('----- Server: get random names start------')
    print(request.headers)
    print(request.form)
    print(type(request.json))
    print(request.json)
    return get_names(request.json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010,debug=True)