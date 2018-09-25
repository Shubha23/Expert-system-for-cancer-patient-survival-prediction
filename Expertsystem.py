'''
#-------------An Expert System for prediction of a patient's survival after undergoing -----
#------------- cancer surgery in last 5 years --------------------------------------------
'''
import numpy as np
import pandas as pd
import json

def get_rules():
    rule_file = 'KBS/knowledgebase.json'   # Rules retrieved from Json file....
    with open(rule_file, 'r') as f:
        rules = json.load(f)
        return rules

def get_facts():
    facts = {'Young', 'Adult', 'Senior','node1', 'node2','node3','node4','node5','node6',
             'node7','node8','node9','node10','year'}
    return facts
    
def match_rule(facts, rule):          # Return true if matched else return false....
    for condition in rule['IF']:
        if (condition not in facts):
            return False
    return True    

def get_testcases():                   # Read the Haberman's dataset 
    cases_file = pd.read_csv('Hbdata.csv')
    age = np.array(cases_file)[:,0]       
    year = np.array(cases_file)[:,1]
    aux_nodes = np.array(cases_file)[:,2]
    status = np.array(cases_file)[:,3]
    return age,year,aux_nodes,status
    
def testing(rules, facts):            # Test each instance by matching with all rules from
    status = set()                    # Knowledge base. Reteive conclusion of one that matches..
    for rule in rules:
        if (match_rule(facts, rule)):
            print('Match rule: ' + str(rule['IF']) + str(rule['THEN']))
            conclusions = rule['THEN']            
            if 'Survival Status' in conclusions.keys():
                status = status | set(conclusions['Survival Status'])
    return status

if __name__ == "__main__":
     rules = get_rules()
     facts = get_facts()
    
     a,y,aux,cases = get_testcases()
     x = []
     pred = []
     acc = []
     actual = []
     
     for i in range(len(cases)):         # Testing on entire dataset (306 instances)...
        if((a[i]>29) and (a[i]<46)):
            f = "Young"
        elif(a[i]>=46 and a[i]<=60):
            f = "Adult"
        elif(a[i]>60):   
            f = "Senior"
        if(aux[i] <= 5):
            n = "node1"
        elif(aux[i]>= 6 and aux[i]<= 10):
            n = "node2"    
        elif(aux[i]>=11 and aux[i]<=20):   
            n = "node3"  
        elif(aux[i]<=21 and aux[i]<=25):
            n = "node4"
        elif(aux[i]<= 26 and aux[i]<=30):
            n = "node5"
        elif(aux[i]>=31 and aux[i]<=35):   
            n = "node6"
        elif(aux[i]>=36 and aux[i]<= 40): 
            n = "node7" 
        elif(aux[i]<=41 and aux[i]<= 45):
            n = "node8"
        elif(aux[i]<= 46 and aux[i]<= 50):
            n = "node8"    
        elif(aux[i]>= 50): 
            n = "node10"     
        if((y[i]>=55) and (a[i]<=70)):
            yr = "year"
        fact = f+n+yr
        facts = fact
        if(cases[i] == 1):
            case = "Yes"
        elif(cases[i] == 2):
            case = "No"
        conclusion = case
        status = testing(rules, facts)
        if len(status) > 0:
            print('Survival Status: ' + str(status))            
        x.append(status)

 #####------------------------------ END --------------------------------------------------##### 