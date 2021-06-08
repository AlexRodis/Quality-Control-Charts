import pandas as pd
import matplotlib.pyplot as plt
import markdown
import numpy as np
from collections import deque,namedtuple
from statistics import stdev, mean
from dataclasses import dataclass
from copy import deepcopy





plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class MovingMean:
    
    Vector = list[float]
    
    def __init__(self, start_pts : Vector, new_pts : Vector)->None:
        '''
        Base class constructor.
        Args:
            start_pts: list[float] A list of N initial measurements without indices
            new_pts: list[float] A list of subsequent measurements without indices

        returns:
            None

        '''

        self.active_pts = deque(start_pts)
        self._cnt = len(self.active_pts)
        self.new_pts = deque(new_pts)
        self.new_pts_cp = deepcopy(self.new_pts)
        self._upper_bound = len(start_pts) + len(start_pts)
        self.lower_bound = len(start_pts)
        self.past_pts = deque()
        self.params = {"mu":[], "sd":[]}
        self._line_data = pd.DataFrame(data = {
            'upper_action':[],
            'lower_action':[],
            'upper_warning':[],
            'lower_warning':[],
            'average':[],
            'idx':[]
        })
        self.__init_table()
        
        
    def __init_table(self)->None:
        '''
        Reshapes self._line_data which contains y coords for all horizontal decision lines by 
        adding column names
        Args:
            self
        Returns:
            None
        '''
        self._line_data.columns = ["upper_action","lower_action","upper_warning","lower_warning","average","idx"]


    def __reshape_table(self)->None:
        '''
        Reshapes table by declaring column idx as index

        Args:
            self

        Return
            None
        '''
        self._line_data = self._line_data.astype({"idx":"int32"})
        self._line_data = self._line_data.set_index('idx')

    def __set_table(self,graph_vars : namedtuple)->None:
        '''
        Set all decision line points for decision lines

        Args
            self
            graph_vars: namedtuple[Vars, [mu,sd]] A named tuple with pairs of [mu,sd] for each record

        Returns:
            None
        '''
        new_vars = {
            "upper_action":graph_vars.mu+3*graph_vars.sd,
            "lower_action":graph_vars.mu-3*graph_vars.sd,
            "upper_warning": graph_vars.mu+2*graph_vars.sd,
            "lower_warning": graph_vars.mu-2*graph_vars.sd,
            "average":graph_vars.mu,
            "idx":self._cnt
        }
        self._line_data = self._line_data.append(new_vars,ignore_index=True)
        
    def __bitshift(self)->None:
        '''
        Pop the oldest point for active, pop the youngest point in self.new_pts and insert it
        into self.active. Deprecated
        Args
            self
        Returns
            None
        '''
        self.active_pts.append(self.new_pts.pop())
        self.past_pts.append(self.active_pts.popleft())
    
    def __get_params(self)->namedtuple:
        '''
        Compute sd and mu parameters for each new observation and add them to a list

        Args
            self
        
        Returns
            namedtuple[Vars, [mu,sd]] A namedtuple of mu,sd pairs
        '''
        vars = namedtuple("Vars", ['mu','sd'])
        c = self.active_pts
        m = mean(c)
        sd = stdev(c)
        self.params["mu"].append(m)
        self.params["sd"].append(sd)      
        vars.mu = m
        vars.sd = sd
        return vars



    def __render__(self)->None:
        '''
        Method called to render final figure

        Args
            self
        
        Returns
            None
        
        Exports
            Matplotlib.figure The QC Plot
        '''
        # This needs to be broken up to a sepperate function/class/object to add basic parameters to the diagram,
        # keeping only the basic border lines here
        fig,ax = plt.subplots()
        ax.set_xlim(self.lower_bound,self._upper_bound)
        for idx,col in self._line_data.iterrows():
            if idx != 20:
                ax.plot((idx-1,idx),(self._line_data.loc[idx]["upper_action"], self._line_data.loc[idx]["upper_action"]),"r--")
                ax.plot((idx-1,idx-1),(self._line_data.loc[idx-1]["upper_action"],self._line_data.loc[idx]["upper_action"]),"r--")
                ax.plot((idx-1,idx),(self._line_data.loc[idx]["lower_action"], self._line_data.loc[idx]["lower_action"]),"r--")
                ax.plot((idx-1,idx-1),(self._line_data.loc[idx-1]["lower_action"],self._line_data.loc[idx]["lower_action"]),"r--")
                
                ax.plot((idx-1,idx),(self._line_data.loc[idx]["upper_warning"], self._line_data.loc[idx]["upper_warning"]),"y--")
                ax.plot((idx-1,idx-1),(self._line_data.loc[idx-1]["upper_warning"],self._line_data.loc[idx]["upper_warning"]),"y--")
                ax.plot((idx-1,idx),(self._line_data.loc[idx]["lower_warning"], self._line_data.loc[idx]["lower_warning"]),"y--")
                ax.plot((idx-1,idx-1),(self._line_data.loc[idx-1]["lower_warning"],self._line_data.loc[idx]["lower_warning"]),"y--")

                ax.plot((idx-1,idx),(self._line_data.loc[idx]["average"],self._line_data.loc[idx]["average"]), "b-")
                ax.plot((idx-1,idx-1),(self._line_data.loc[idx-1]["average"], self._line_data.loc[idx]["average"]), "b-")

        
        ax.set_xlabel("Αριθμός Μέτρησης")
        ax.set_ylabel("Μέτρηση")
        ax.set_xticks(np.arange(20,55,step = 5))
        ax.set_title("Διάγραμμα QC Κινούμενου Μέσου Όρου")
        ax.plot(np.arange(20,len(self.new_pts_cp) + 20,step=1), (self.new_pts_cp), "kP")
        plt.show()
                
            
            
            
    def __call__(self)->None:
        '''
        Once basic parameters are set, call the object to initiate computations

        Args
            self
        Returns
            None
        '''
        # This may need to be broken up to compute and render steps
        graph_vars = self.__get_params()
        self.__set_table(graph_vars)
        new_pts = deepcopy(self.new_pts)
        for elem in new_pts:
            self._cnt += 1
            self.__bitshift()
            self.__set_table(self.__get_params())
        self.__reshape_table()
        self.__render__()

x = pd.read_csv("Διαγράματα Ελέγχου Ποιότητας Τ.Α.1 Μέσου Όρου.csv",index_col=False)
y = pd.read_csv("Διαγράματα Ελέγχου Ποιότητας Τ.Α.2 Εκτός Ελέγχου.csv",index_col=False)


x = MovingMean(list(x["Result"]),list(y["Result"]))
x()