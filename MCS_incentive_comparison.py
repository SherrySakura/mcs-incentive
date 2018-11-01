# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:30:48 2018

@author: pc
"""



import numpy as np
import random
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid",{'axes.edgecolor': '.0', 'axes.linewidth': 1.0,})
sns.set_context("notebook", font_scale = 1.5)


r_original = 5
r_original = 3

'''
this test evaluates the performance of three methods, our bonus-based mechanism, greedy mechanism and uniform incentive mechanism,
in terms of user utility, data quality, and participant proportion
when the number of tasks increases
'''

# two types
type_of_task = [0,1]
# origianl reward of each type
r=[4,4]

#type of each task                        
task_type_standard=[1,0,0,0,0,0,0,1]
#data quality of each users to execute two types of tasks
q_user_tasktype_standard=[[10,5],[9,3],[10,7],[9,3],[10,4],[9,3],[10,5],[7,6]]
#cost of each users to execute two types of tasks
c_user_tasktype_standard=[[1.4,1.2],[1.3,1.0],[2.5,1.8],[3.4,3.1],[2.9,4],[4.0,2.9],[2,5],[2.1,2]]

                         
u=[]
k=[]
q=[]
c=[]
rr=[]


preference_user=[]

cost_coefficience =0.3

def gain_utility_quality_and_cost():    
    for i in range(0,len(q_user_tasktype)):#each user in the loop to obtain their utility
        k=[]
        kk=[]
        kkk=[]
        rrr=[]
        for j in range(0,len(task_type)):
            summm=q_user_tasktype[i][task_type[j]]*cost_coefficience+r[task_type[j]]-c_user_tasktype[i][task_type[j]]
            summ_q=q_user_tasktype[i][task_type[j]]
            summ_c=c_user_tasktype[i][task_type[j]]
            rrrr=r[task_type[j]]
            k.append(summm)
            kk.append(summ_q)
            kkk.append(summ_c)
            rrr.append(rrrr)
        u.append(k)
        q.append(kk)
        c.append(kkk)
        rr.append(rrr)
    for i in range(0,len(q_user_tasktype)):
        for j in range(0,len(task_type)):
            if c[i][j]>=r[task_type[j]]:
                u[i][j]=0
    
    '''
    print('the utility of user is:\n',u)
    print()
    print('the data quality of users when executing tasks are:\n',q)
    print()
    print('the cost of users when executing tasks are:\n',c)
    print()
    print('the original reward of users when executing tasks are:\n',rr)
    '''
    return u,q,c


    
    
    
    
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
    #print('the stable matching is:\n',match)
    return match



def preference_construct(u,q): 
    
    preference_user=[]
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
    '''
    print()    
    print('the preference list of user is:\n',preference_user)
    print()
    print('the preference list of task is:\n',preference_task)
    print()
    '''
    return preference_user, preference_task




def task_user(u,q):
    preference_user, preference_task = preference_construct(u,q)
    MP = preference_user
    WP = preference_task
    SM = find_stable_matching(MP,WP)
    utility_i = np.sum(u[SM[k][0]][SM[k][1]] for k in range(0,len(u)))
    utility_j = np.sum(q[SM[k][0]][SM[k][1]] for k in range(0,len(q)))
    bonus_total = utility_j*cost_coefficience
    money_total = bonus_total + len(MP)*r[0]
    money=[]
    u_ave=[]
    a=0
    for i in range(0,len(WP)):
        money.append(money_total/len(WP))
        if c[i][SM[i][1]]<r[task_type[SM[i][0]]]:
            a+=1
    rate=a/len(WP)
    
    for i in range(0,len(q_user_tasktype)):
        k=[]
        for j in range(0,len(task_type)):
            summm=money[i]-c_user_tasktype[i][task_type[j]]
            k.append(summm)
        u_ave.append(k)
        
        
    '''
    print('the utility of user is:\n',u)
    print()
    print('the overall utility of users are:\n',utility_i)
    print()
    print('the total quality of tasks are:\n',utility_j)
    print()
    print('the total money of tasks are:\n',money_total)
    print()
    print('the average distribution of money, each user can gains:\n',money)
    print()
    print('the utility of users when money average distriuted:\n',u)
    '''
    return utility_i,utility_j,rate,money_total,money,u_ave



def task_filtering(c,r):
    '''
    for each user, filter the tasks, only hold the tasks users can gain reward.
    '''
    num_user = len(c)
    num_task = len(c[0])    
    user_execute_or_not_list=[([-1] * num_task) for i in range(num_user)]
    #print('initial execute or not list:',user_execute_or_not_list)
    #print()
    '''
    -1 represents user i not necessary to execute task j
    1 represents user i can execute task j
    '''
    for i in range(0,num_user):
        num_task = len(c[i])
        for j in range(0,num_task):
            if c[i][j]<r[task_type[j]]:
                user_execute_or_not_list[i][j]=1
                #print('user',i,'can execute task',j)
                #print()
    #print('user_execute_or_not_list:',user_execute_or_not_list)
    #print()
    
    user_can_execute_tasklist =[([] * num_task) for i in range(num_user)] 
    for i in range(0,num_user):
        for j in range(0,num_task):
            if user_execute_or_not_list[i][j]==1:
                user_can_execute_tasklist[i].append(j)
    #print('user_can_execute_tasklist',user_can_execute_tasklist)     
    #print()
    return user_execute_or_not_list,user_can_execute_tasklist
   
def task_random_filter():
    n=len(q)
    user=[]
    #task=[]
    select_task=[]
    match=[(-1,-1)]*n
    a=0
    for i in range(0,n):
        user.append(i)
    user_execute_or_not_list,user_can_execute_tasklist = task_filtering(c,r)
    #print('user_can_execute_tasklist',user_can_execute_tasklist)
    for i in range(0,n):
        for k in range(0,100):
            select = random.choice(user_can_execute_tasklist[i])
            if select not in select_task:
                select_task.append(select)
                a+=1
                break
    rate=a/n
    #print(select_task)        
    #random.shuffle(task)
    for i in range(a):
        match[i]=(user[i],select_task[i])
    for i in range(0,len(match)):
        if match[i]==(-1,-1):
            u[i][match[i][1]]=0
    utility_i = np.sum(u[match[k][0]][match[k][1]] for k in range(0,len(u)))
    utility_j = np.sum(q[match[k][0]][match[k][1]] for k in range(0,len(q)))
    #print('match:\n',match)
    #print('random case with bonus, the overall utility of users are:\n',utility_i)
    #print()
    #print('random case with bonus, the total quality of tasks are:\n',utility_j)
    #print() 
    return utility_i,utility_j,rate,match

def task_random():
    n=len(q)
    user=[]
    task=[]
    match=[(-1,-1)]*n
    u_random=[]
    q_random=[]
    utility_i=0
    utility_j=0
    a=0
    
    for i in range(0,n):
        user.append(i)
    for i in range(0,n):
        task.append(i)        
    random.shuffle(task)
    for i in range(len(q)):
        match[i]=(user[i],task[i])
        u_random.append(u[match[i][0]][match[i][1]])
        q_random.append(q[match[i][0]][match[i][1]])
    #utility_i = np.sum(u[match[k][0]][match[k][1]] for k in range(0,len(u)))
    for k in range(0,n):
        if c[k][match[k][1]]<r[task_type[match[k][1]]]:
            utility_j += q[match[k][0]][match[k][1]]
            utility_i += u[match[k][0]][match[k][1]]
            a+=1
    rate=a/n
    '''
    print('random case, the overall utility of users are:\n',utility_i)
    print()
    print('random case, the total quality of tasks are:\n',utility_j)
    print()
    print('random case, the utility of users are:\n',u_random)
    print()    
    print('random case, the quality of tasks are:\n',q_random)
    print() 
    '''
    return utility_i,utility_j,rate,match
    

def task_greedy():
    n=len(q)
    user=[]
    task=[]
    match=[(-1,-1)]*n
    
    
    utility_i=0
    utility_j=0
    a=0
    
    for i in range(0,n):
        user.append(i)
    for i in range(0,n):
        task.append(i)
    #print(task)
    task_remain=task.copy()
    for i in range(0,n):
        s=-10
        for j in task_remain:
            if s<u[i][j]:
                s=u[i][j]
                task_select=j
        match[i]=(user[i],task_select)
        task_remain.remove(task_select)        
        #print(match[i])
        #print('delete:',task_select)
        

    for k in range(0,n):
        if c[k][match[k][1]]<r[task_type[match[k][1]]]:
            utility_j += q[match[k][0]][match[k][1]]
            utility_i += u[match[k][0]][match[k][1]]
            a+=1
    rate=a/n
    '''
    print('random case, the overall utility of users are:\n',utility_i)
    print()
    print('random case, the total quality of tasks are:\n',utility_j)
    print()
    print('random case, the utility of users are:\n',u_random)
    print()    
    print('random case, the quality of tasks are:\n',q_random)
    print() 
    '''
    return utility_i,utility_j,rate,match



utility_j_Collection = []
utility_j_2_Collection = []
utility_j_ave_Collection = []
utility_j1_ave_Collection = []
utility_j_greedy_Collection=[]

utility_i_Collection=[]
utility_i_2_Collection=[]
utility_i_ave_Collection=[]
utility_i1_ave_Collection=[]
utility_i_greedy_Collection=[]



rate_my_Collection=[]
rate_eq_Collection=[]
rate_ave_random_with_bonus=[]
rate_ave_random_without_bonus=[]
rat_greedye_Collection=[]
    
if __name__=="__main__":
    for count in range(1,13):
        task_type = task_type_standard*count
        q_user_tasktype = q_user_tasktype_standard*count
        c_user_tasktype = c_user_tasktype_standard*count
        preference_user=[]
        cost_coefficience =0.3
        u=[]
        k=[]
        q=[]
        c=[]
        rr=[]
        type_of_task = [0,1]
        r=[4,4]#两种任务的基本工资
        '''
        print('when the total number of users is:\n',len(task_type))
        print('when the total number of task types is:\n',len(type_of_task))
        print('when the rate of two tasks are:\n',task_type.count(0)/task_type.count(1))
        '''
        utility_i_temp=[]
        utility_j_temp=[]
        utility_i1_temp=[]
        utility_j1_temp=[]    
        rate_temp=[]
        rate1_temp=[]
        u,q,c=gain_utility_quality_and_cost()
        user_execute_or_not_list,user_can_execute_tasklist = task_filtering(c,r)
        #print('user_can_execute_tasklist',user_can_execute_tasklist)
        '''
        print("**********introduce reward/bonus************")
        dynamic_learning()
        print('**********without reward/bonus*************')
        dynamic_learning_without_filtering()
        '''
    
    
        print('**********user-task matching*************')
        utility_i,utility_j,rate_my,money_total,money,u_average=task_user(u,q)
        '''
        print('the overall utility of users are:\n',utility_i)
        
        print('the total quality of tasks are:\n',utility_j)
        
        print('the rate of users success matching:\n',rate_my)
        '''
        utility_i_Collection.append(utility_i)
        utility_j_Collection.append(utility_j)
        rate_my_Collection.append(rate_my)
    
    
        print('**********equal reward, user-task matching*************')
        utility_i_2,utility_j_2,rate_eq,money_total_2,money_2,u_average_2=task_user(u_average,q)
        
        utility_i_2_Collection.append(utility_i_2)
        
        utility_j_2_Collection.append(utility_j_2)
        
        rate_eq_Collection.append(rate_eq)
        '''
        print('without bonus, the overall utility of users are:\n',utility_i_2)
        print('without bonus, the total quality of tasks are:\n',utility_j_2)
        print('equal reward, the rate of users success matching:\n',rate_eq)
        #print('total input of requester:\n',sum(money))
        if sum(money)>utility_j:
            print('oops! cost too much for requester!')
        '''    
        print('**********user-task greedy*************')
        utility_i_greedy,utility_j_greedy,rat_greedye,match_greedy=task_greedy()
        utility_i_greedy_Collection.append(utility_i_greedy)
        utility_j_greedy_Collection.append(utility_j_greedy)
        rat_greedye_Collection.append(rat_greedye)
        
        print('**********user-task random without matching, with bonus*************')
        for i in range(0,200):
            utility_i,utility_j,rate,match=task_random_filter()
            utility_i_temp.append(utility_i)
            utility_j_temp.append(utility_j)
            rate_temp.append(rate)
            #print('match:\n',match)
        utility_i_ave=sum(utility_i_temp)/len(utility_i_temp)
        utility_j_ave=sum(utility_j_temp)/len(utility_j_temp)
        rate_ave=sum(rate_temp)/len(rate_temp)
        
        utility_i_ave_Collection.append(utility_j_ave)
        
        utility_j_ave_Collection.append(utility_j_ave)
        
        
        rate_ave_random_with_bonus.append(rate_ave)
        '''
        print('average total utility in 200 times is:\n',utility_i_ave)
        print('average total quality in 200 times is:\n',utility_j_ave)
        print('the rate of users success matching:\n',rate_ave)
        print('match:\n',match)
        '''
        print('**********user-task random without matching, without bonus*************')
        for i in range(0,200):
            utility_i1,utility_j1,rate1,match1=task_random()
            utility_i1_temp.append(utility_i)
            utility_j1_temp.append(utility_j)
            rate1_temp.append(rate1)
            #print('match:\n',match)
        utility_i1_ave=sum(utility_i1_temp)/len(utility_i1_temp)
        utility_j1_ave=sum(utility_j1_temp)/len(utility_j1_temp)
        rate1_ave=sum(rate1_temp)/len(rate1_temp)
        
        
        utility_i1_ave_Collection.append(utility_i1_ave)
        utility_j1_ave_Collection.append(utility_j1_ave)
        
        
        rate_ave_random_without_bonus.append(rate1_ave)
        
        '''
        print('average total utility in 200 times is:\n',utility_i1_ave)
        print('average total quality in 200 times is:\n',utility_j1_ave)
        print('the rate of users success matching:\n',rate1_ave)
        print('match:\n',match1)
        '''
        '''
        #task_random_filter()
        
        '''
    
    print()    
    
    print("utility_j_Collection:")
    print(utility_j_Collection)
    print("utility_j_2_Collection:")
    print(utility_j_2_Collection)
    print("utility_j_ave_Collection:")
    print(utility_j_ave_Collection)
    print("utility_j1_ave_Collection")
    print(utility_j1_ave_Collection)
    print("utility_j_greedy_Collection")
    print(utility_j_greedy_Collection)
    

    print()

    print("utility_i_Collection:")
    print(utility_i_Collection)
    print("utility_i_2_Collection:")
    print(utility_i_2_Collection)
    print("utility_i_ave_Collection:")
    print(utility_i_ave_Collection)
    print("utility_i1_ave_Collection")
    print(utility_i1_ave_Collection)
    print("utility_i_greedy_Collection")
    print(utility_i_greedy_Collection)
    
    print()

    print("rate_my_Collection:")
    print(rate_my_Collection)
    print("rate_eq_Collection:")
    print(rate_eq_Collection)
    print("rate_ave_random_with_bonus:")
    print(rate_ave_random_with_bonus)
    print("rate_ave_random_without_bonus")
    print(rate_ave_random_without_bonus)
    print("rat_greedye_Collection")
    print(rat_greedye_Collection)
    
 
'''
in the following, show the numerical results
'''
  
'''
fig1=plt.figure(1)

a=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
plt.plot(utility_j_Collection,'g-',color='coral',linewidth=2,markeredgewidth=2.4,marker='o',markerfacecolor='w',markeredgecolor='coral',alpha=1,label='Bonus-based incentive mechanism')
plt.plot(utility_j_greedy_Collection,'g-.',color='#33b864',linewidth=2,markeredgewidth=2.4,marker='^',markerfacecolor='w',markeredgecolor='#33b864',alpha=1,label='Greedy algorithm')
plt.plot(utility_j_ave_Collection,'g--',color='#3e82fc',linewidth=2,markeredgewidth=2.4,marker='x',markerfacecolor='w',markeredgecolor='#3e82fc',alpha=1, label='Random task allocation, with bonus')
#plt.plot(utility_j1_ave_Collection,'g:',color='#db4bda',linewidth=2,markeredgewidth=2.4,marker='o',markerfacecolor='w',markeredgecolor='#db4bda',alpha=1, label='Random task allocation, without bonus')
plt.legend(loc='upper center', bbox_to_anchor=(0.37,1.0),ncol=1,fancybox=True,shadow=True)
plt.xlabel('Number of users')
plt.ylabel('Total data quality of users')
plt.xticks(a, ('8','16','24','32','40','48','56','64','72','80','88','96') )
plt.xlim(-0.2, 7.1)
plt.ylim(min(min(utility_j_Collection),min(utility_j_2_Collection),min(utility_j_ave_Collection),min(utility_j1_ave_Collection)) * 0.9, max(max(utility_j_Collection),max(utility_j_2_Collection),max(utility_j_ave_Collection),max(utility_j1_ave_Collection)) * 1.4)
plt.ylim(0,max(max(utility_j_Collection),max(utility_j_2_Collection),max(utility_j_ave_Collection),max(utility_j1_ave_Collection)) * 1.05)
plt.ylim(0,600)

#plt.grid(False)
plt.show()
fig1.savefig("total_quality_different_user_num.png", dpi=1000)

fig2=plt.figure(2)

a=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
plt.plot(rate_my_Collection,'g-',color='coral',linewidth=2,markeredgewidth=2.4,marker='o',markerfacecolor='w',markeredgecolor='coral',alpha=1,label='Bonus-based incentive mechanism')
plt.plot(rat_greedye_Collection,'g-.',color='#33b864',linewidth=2,markeredgewidth=2.4,marker='^',markerfacecolor='w',markeredgecolor='#33b864',alpha=1,label='Greedy mechanism')
plt.plot(rate_ave_random_with_bonus,'g--',color='#3e82fc',linewidth=2,markeredgewidth=2.4,marker='x',markerfacecolor='w',markeredgecolor='#3e82fc',alpha=1, label='Random task allocation, with bonus')
plt.plot(rate_eq_Collection,'g:',color='#db4bda',linewidth=2,markeredgewidth=2.4,marker='o',markerfacecolor='w',markeredgecolor='#db4bda',alpha=1, label='Uniform incentive mechanism')
plt.legend(loc='upper center', bbox_to_anchor=(0.67,0.5),ncol=1,fancybox=True,shadow=True)
plt.xlabel('Number of users')
plt.ylabel('Total data quality of users')
plt.xticks(a, ('8','16','24','32','40','48','56','64','72','80','88','96') )
plt.xlim(-0.2, 11.1)
#plt.ylim(min(min(rate_my_Collection),min(rat_greedye_Collection),min(rate_ave_random_with_bonus),min(rate_eq_Collection)) * 0.9, max(max(rate_my_Collection),max(rat_greedye_Collection),max(rate_ave_random_with_bonus),max(rate_eq_Collection)) * 1.2)
plt.ylim(0,max(max(rate_my_Collection),max(rat_greedye_Collection),max(rate_ave_random_with_bonus),max(rate_eq_Collection)) * 1.05)

#plt.grid(False)
plt.show()
fig2.savefig("rate_different_user_num.png", dpi=1000)
'''






myfont = matplotlib.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size = 10)
hatch_s=['xx','||','\\\\','|','+','/','|','.','*','O','xx','///']

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()-0.19, 1.01*height, '%s' % float(height),fontsize=13)

#rect represents the number of bar colors

n_groups = 3

bar_width = 0.2

def round_2(a):
    b=[]
    for i in a:
        b.append(round(i,2))
    return b
a=round_2(rate_my_Collection[:3])
b=round_2(rat_greedye_Collection[:3])
c=round_2(rate_eq_Collection[:3])

MAE0 = np.array(a)
RC0 = np.array(b)
OP0 = np.array(c)
rects0 = [0.]*3

index = np.arange(0,n_groups)


plt.figure(4,figsize=(7,5))
rects0[0]= plt.bar(index, MAE0, bar_width,edgecolor = 'coral', label = "Bonus-based incentive mechanism", color = 'coral')
rects0[1]= plt.bar(index+bar_width, RC0, bar_width,edgecolor = '#7efbb3', label = "Greedy mechanism", color ='#7efbb3')
rects0[2]= plt.bar(index+bar_width+bar_width, OP0,bar_width,edgecolor = '#3e82fc', label= 'Uniform incentive mechanism', color = '#3e82fc')
plt.ylabel('Proportion of participants', fontsize=14)  
plt.yticks(fontsize=14)
plt.xticks(index + bar_width*1.04, ('8 tasks','16 tasks','24 tasks'),fontsize= 14)
plt.ylim(0.8,1.1)
plt.legend(loc='best', ncol=1,fontsize= 14)
plt.tight_layout()  
for i in range(len(rects0)):
    autolabel(rects0[i])
plt.savefig("Compare.jpg", dpi=300, bbox_inches='tight')
plt.show()


p1=np.array(round_2(utility_j_Collection[:3]))
p2=np.array(round_2(utility_j_greedy_Collection[:3]))
p3=np.array(round_2(utility_j_2_Collection[:3]))
plt.figure(5,figsize=(7,5))
rects0[0]= plt.bar(index, p1, bar_width,edgecolor = 'coral', label = "Bonus-based incentive mechanism", color = 'coral')
rects0[1]= plt.bar(index+bar_width, p2, bar_width,edgecolor = '#7efbb3', label = "Greedy mechanism", color ='#7efbb3')
rects0[2]= plt.bar(index+bar_width+bar_width, p3,bar_width,edgecolor = '#3e82fc', label= 'Uniform incentive mechanism', color = '#3e82fc')
plt.ylabel('Total data quality from users', fontsize=14)  
plt.yticks(fontsize=14)
plt.xticks(index + bar_width*1.04, ('8 tasks','16 tasks','24 tasks'),fontsize= 14)
plt.ylim(0,220)
plt.legend(loc='best', ncol=1,fontsize= 14)
plt.tight_layout()  
for i in range(len(rects0)):
    autolabel(rects0[i])
plt.savefig("Compare2.jpg", dpi=300, bbox_inches='tight')
plt.show()


p4=np.array(round_2(utility_i_Collection[:3]))
p5=np.array(round_2(utility_i_greedy_Collection[:3]))
p6=np.array(round_2(utility_i_2_Collection[:3]))
plt.figure(6,figsize=(7,5))
rects0[0]= plt.bar(index, p4, bar_width,edgecolor = 'coral', label = "Bonus-based incentive mechanism", color = 'coral')
rects0[1]= plt.bar(index+bar_width, p5, bar_width,edgecolor = '#7efbb3', label = "Greedy mechanism", color ='#7efbb3')
rects0[2]= plt.bar(index+bar_width+bar_width, p6,bar_width,edgecolor = '#3e82fc', label= 'Uniform incentive mechanism', color = '#3e82fc')
plt.ylabel('Total utilities of users', fontsize=14)  
plt.yticks(fontsize=14)
plt.xticks(index + bar_width*1.04, ('8 tasks','16 tasks','24 tasks'),fontsize= 14)
plt.ylim(0,120)
plt.legend(loc='best', ncol=1,fontsize= 14)
plt.tight_layout()  
for i in range(len(rects0)):
    autolabel(rects0[i])
plt.savefig("Compare3.jpg", dpi=300, bbox_inches='tight')
plt.show()