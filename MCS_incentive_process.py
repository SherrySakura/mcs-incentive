# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:40:02 2018

@author: pc
"""

import numpy as np
import random
import copy

q=[[10,10,4,4],[9,11,5,4],[5,4,9,10],[4,5,10,9]]
c=[[1,1,5,8],[2,1,3,4],[6,2,1,2],[4,6,2,3]]
r=[4,4,4,4]
u=[]
k=[]
type_task=[0,0,1,1]
preference_user=[]
for i in range(0,len(q)):
    k=[]
    for j in range(0,len(q[i])):
        summm=q[i][j]*0.2+4-c[i][j]
        k.append(summm)
    u.append(k)
print('the utility of user is:\n',u)


def task_filtering(c,r):
    '''
    for each user, filter the tasks, only hold the tasks users can gain reward.
    '''
    num_user = len(c)
    num_task = len(c[0])    
    user_execute_or_not_list=[([-1] * num_task) for i in range(num_user)]
    print('initial execute or not list:',user_execute_or_not_list)
    print()
    '''
    -1 represents user i not necessary to execute task j
    1 represents user i can execute task j
    '''
    for i in range(0,num_user):
        num_task = len(c[i])
        for j in range(0,num_task):
            if c[i][j]<r[j]:
                user_execute_or_not_list[i][j]=1
                print('user',i,'can execute task',j)
                print()
    print('user_execute_or_not_list:',user_execute_or_not_list)
    print()
    
    user_can_execute_tasklist =[([] * num_task) for i in range(num_user)] 
    for i in range(0,num_user):
        for j in range(0,num_task):
            if user_execute_or_not_list[i][j]==1:
                user_can_execute_tasklist[i].append(j)
    print('user_can_execute_tasklist',user_can_execute_tasklist)     
    print()
    return user_execute_or_not_list,user_can_execute_tasklist

#each user learn the perference of task types when a new task comes
def preference_learning(q,c,r,u,type_task):
    '''
    each user learns it reward through trying.
    q[i] encodes the ith user's quality. It is a list containing the quality of executing each task.
    c[i] encodes the ith user's cost. It is a list containing the cost of executing each task.
    r is a list containing the original reward of each task.
    '''
    sum_q=0
    try_time=100
    num_task=len(c)
    num_user=len(c[0])
    episode = 1.0
    expected_q=[([-1000] * num_task) for i in range(num_user)]
    expected_u=[([-1000] * num_task) for i in range(num_user)]
    every_episode_u=[[] for i in range(try_time)]
    every_episode_q=[[] for i in range(try_time)]
    every_episode_u_ave=[]
    every_episode_q_ave=[]
#    prefer_task=[([] * num_task) for i in range(num_user)]
    
    #select the tasks each user can execute
    #user_execute_or_not_list is a 2-D list, where 1 represents user executes task, -1 represents user not execute task.
    #user_can_execute_tasklist is a 2-D list, where i th user with [1,2,3] represents it can execute task 1,2,3 
    user_execute_or_not_list,user_can_execute_tasklist = task_filtering(c,r)
    
    for try_round in range(0,try_time):
        for i in range(0,num_user):
            ret = random.random()
            # with high possibility 80%
            if ret < episode+1 or try_time < 10:
                pick_task = user_can_execute_tasklist[i][random.randint(0,len(user_can_execute_tasklist[i])-1)]
                expected_q[i][pick_task] = q[i][pick_task]
                expected_u[i][pick_task] = u[i][pick_task]
                print('user',i,':quality of selected task',pick_task,'is',expected_q[i][pick_task])
                print('user',i,':utility of selected task',pick_task,'is',expected_u[i][pick_task])
                temp_q = expected_q[i][pick_task]
                temp_u = expected_u[i][pick_task]
                #print('the random picked task of user',i,'is',pick_task)
            every_episode_q[try_round].append(temp_q)
            every_episode_u[try_round].append(temp_u)
            print('current round',try_round,'quality:',every_episode_q[try_round])
            print('current round',try_round,'quality:',every_episode_u[try_round])
        every_episode_u_ave.append((sum(every_episode_u[try_round][i] for i in range(0,num_user)))/len(every_episode_u[try_round]))
        every_episode_q_ave.append((sum(every_episode_q[try_round][i] for i in range(0,num_user)))/len(every_episode_q[try_round]))
                #print()
        for i in range(0,num_user):
            sum_q += sum(expected_q[i][j] for j in user_can_execute_tasklist[i])
        if sum_q>0:
            break
           
    return expected_q,expected_u,every_episode_q_ave,every_episode_u_ave
    
    
    
    
def find_stable_matching(MP,WP):
    """ The input of this function is defined by two lists MP and WP
    Men and women are numbered 0 through n-1
    MP[i] encodes the ith man's preferences. It is simply a list
    containing the numbers 0, 1, ... , n-1 in some order
    WP[i] encodes the ith woman's preferences. It is simply a list
    containing the numbers 0, 1,... , n-1 in some order
    The output is defined by a list of pairs of the form (i,j)
    indicating that man i is married to woman j
    Your output should be the man-optimal stable matching found by the
    GS-algorithm. """
     # your code goes here!
    n=len(MP)
    isManFree=[True]*n
    isWomanFree=[True]*n
    #isManProposed=[[False]*n]*n
    isManProposed=[[False for i in range(n)] for j in range(n)]
    match=[(-1,-1)]*n
 
 
    while(True in isManFree):
        indexM=isManFree.index(True)
        if(False in isManProposed[indexM]):
            indexW=-1
            for i in range(n):
                w=MP[indexM][i]
                if(not isManProposed[indexM][w]):
                    indexW=w
                    break
            isManProposed[indexM][indexW]=True
            if(isWomanFree[indexW]):
                #isManProposed[indexM][indexW]=True
                isWomanFree[indexW]=False
                isManFree[indexM]=False
                match[indexM]=(indexM,indexW)
            else:
                indexM1=-1
                for j in range(n):
                    if(match[j][1]==indexW):
                        indexM1=j
                        break
                if(WP[indexW].index(indexM)<WP[indexW].index(indexM1)):
                    isManFree[indexM1]=True
                    isManFree[indexM]=False
                    match[indexM]=(indexM,indexW)
    print('the stable matching is:\n',match)
    return match


def dynamic_learning():
    expected_q,expected_u,every_episode_q_ave,every_episode_u_ave=preference_learning(q,c,r,u,type_task)
    print('expected quality of each user:',expected_q)
    print()
    print('expected utility of each user:',expected_u)
    print()
    print('average quality of each episode:',every_episode_q_ave)
    print()
    print('average utility of each episode:',every_episode_u_ave)
    print()
    
    
sort_u = copy.deepcopy(u)
for i in range(0,len(sort_u)):
    preference_i=[]
    for j in range(0,len(sort_u[i])):
        max_u = sort_u[i].index(max(sort_u[i]))
        sort_u[i][max_u]=-1000
        preference_i.append(max_u)
    preference_user.append(preference_i)
    
preference_task=[]
sort_q = copy.deepcopy(q)
for i in range(0,len(sort_q)):
    preference_j=[]
    for j in range(0,len(sort_q[i])):
        max_q = sort_q[i].index(max(sort_q[i]))
        sort_q[i][max_q]=-1000
        preference_j.append(max_q)
    preference_task.append(preference_j)
print()    
print('the preference list of user is:\n',preference_user)
print()
print('the preference list of task is:\n',preference_task)
print()

def task_user():
    
    MP = preference_user
    WP = preference_task
    SM = find_stable_matching(MP,WP)
    utility_i = np.sum(u[SM[k][0]][SM[k][1]] for k in range(0,len(u)))
    utility_j = np.sum(q[SM[k][0]][SM[k][1]] for k in range(0,len(q)))
    print('the overall utility of users are:\n',utility_i)
    print()
    print('the total quality of tasks are:\n',utility_j)
    print()

def task_random():
    n=len(q)
    user=[]
    task=[]
    match=[(-1,-1)]*n
    for i in range(0,n):
        user.append(i)
    for i in range(0,n):
        task.append(i)        
    random.shuffle(task)
    for i in range(len(q)):
        match[i]=(user[i],task[i])
    utility_i = np.sum(u[match[k][0]][match[k][1]] for k in range(0,len(u)))
    utility_j = np.sum(q[match[k][0]][match[k][1]] for k in range(0,len(q)))
    print('random case, the overall utility of users are:\n',utility_i)
    print()
    print('random case, the total quality of tasks are:\n',utility_j)
    print()    
    
if __name__=="__main__":
    print("**********test1************")
    dynamic_learning()
    task_user()
    task_random()








