import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np


## ---------------------------------------
def clean_negative(dataframe):
    """
    Replace any negative reflectivity and RhoHV values by a NaN
    """

    dataframe[dataframe < 0] = np.nan


## ---------------------------------------
def collapse_time_series(dataframe, clean_neg=False):
    """
    Collapses the time series for each radar reading resulting in a single row
    per radar reading/prediction. 
    """
    
    ref = dataframe[[
            'Id',
            'Ref',
            'Ref_5x5_10th',
            'Ref_5x5_50th',
            'Ref_5x5_90th',
            'RefComposite',
            'RefComposite_5x5_10th',
            'RefComposite_5x5_50th',
            'RefComposite_5x5_90th'
                       ]]
    
    ref_id_groups = ref.groupby('Id', as_index=False)
    ref = ref_id_groups.mean()
    if clean_neg:
        clean_negative(ref)
    
    rhohv = dataframe[[
            'Id',
            'RhoHV',
            'RhoHV_5x5_10th',
            'RhoHV_5x5_50th',
            'RhoHV_5x5_90th'
        ]]
    
    rhohv_id_groups = rhohv.groupby('Id', as_index=False)
    rhohv = rhohv_id_groups.mean()
    if clean_neg:
        clean_negative(rhohv)
    
    zdr = dataframe[[
            'Id',
            'Zdr',
            'Zdr_5x5_10th',
            'Zdr_5x5_50th',
            'Zdr_5x5_90th'
        ]]
    
    zdr_id_groups = zdr.groupby('Id', as_index=False)
    zdr = zdr_id_groups.mean()
    
    kdp = dataframe[[
            'Id',
            'Kdp',
            'Kdp_5x5_10th',
            'Kdp_5x5_50th',
            'Kdp_5x5_90th'
        ]]
    
    kdp_id_groups = kdp.groupby('Id', as_index=False)
    kdp = kdp_id_groups.mean()
    
    other = dataframe[[
            'Id',
            'radardist_km',
            'Expected'
        ]]
    
    other_id_groups = other.groupby('Id', as_index=False)
    other = other_id_groups.aggregate(np.amax)
    
    new_dataframe = pd.merge(ref, rhohv, on='Id')
    new_dataframe = pd.merge(new_dataframe, zdr, on='Id')
    new_dataframe = pd.merge(new_dataframe, kdp, on='Id')
    new_dataframe = pd.merge(new_dataframe, other, on='Id')
    
    return new_dataframe


## ---------------------------------------
def correlation_plot(dataframe, var1, var2, bounds = None):
    """
    produces a 2D histogram for a pair of variables
    """

    dataframe_var1_nonan = dataframe[~dataframe[var1].isnull()]
    dataframe_var2_nonan = dataframe_var1_nonan[~data_var1_nonan[var2].isnull()]
    
    x = dataframe_var2_nonan[var1].values
    y = dataframe_var2_nonan[var2].values

    if bounds is None:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
    else:
        
        (xmin, xmax, ymin, ymax) = bounds

    fig, ax = plt.subplots()

    ax.set_xlabel(var1)
    ax.set_ylabel(var2)

    plt.hexbin(x, y, cmap=plt.cm.YlOrRd_r, gridsize=100, bins='log')
    plt.axis([xmin, xmax, ymin, ymax])

    return plt


## ---------------------------------------
def clean_stuck_gauges(dataframe, threshold=50):
    """
    Remove rain gauge values that occur too often, indicating that those are stuck
    """

    dataframe['count'] = 1
    counts = dataframe.groupby('Expected', as_index=False).count()
    large_counts = counts[counts['count'] > threshold]

    def high_count(expected):
        return len(large_counts[large_counts['Expected'] == expected]) > 0

    dataframe['large_count'] = map(high_count, dataframe['Expected'])
    clean_dataframe = dataframe[dataframe['large_count']]

    clean_dataframe.drop('count', 1)
    clean_dataframe.drop('large_count', 1)

    return clean_dataframe


## ========================================
class ScatterMatrix:
    """
    A class to create a matrix of scatter plots from a pandas data frame
    """
    
    ## ----------------------------------------
    def __init__(self, dataframe):
        """
        Constructor
        """
        
        self.data = dataframe
        
        self.n = 0
        self.variables = []
        self.labels = []
        self.lo = []
        self.hi = []
        
        
    ## ----------------------------------------
    def add(self, variable_name, lo=None, hi=None, label=None):
        """
        Add a variable to the matrix:
        'variable_name' is the name of the variable to plot as it appears in the data frame
        'label' is how the variable name should appear on the plot. If left unspecified, 'variable_name' will be used.
        'lo' is the lower bound to show for this variable. If left unspecifed, the minimum value will be used
        'hi' is the highest bound...
        """
        
        self.variables.append(variable_name)
        
        if label is None:
            label = variable_name
        self.labels.append(label)
        
        if lo is None:
            lo = self.data[variable_name].values.min()
        self.lo.append(lo)
        
        if hi is None:
            hi = self.data[variable_name].values.max()
        self.hi.append(hi)
        
        self.n += 1
        
        
    ## ----------------------------------------
    def plot(self):
        """
        produces the plot
        """
        
        fig, axes = plt.subplots(nrows=self.n, ncols=self.n, figsize=(2*self.n,2*self.n))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        
        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
            # Set background color
            ax.set_axis_bgcolor('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')

            # Set up ticks only on one side for the "edge" subplots...
            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')
                
        # Plot the data.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i,j), (j,i)]:
                axes[x,y].set_ylim([self.lo[y], self.hi[y]])
                axes[x,y].set_xlim([self.lo[x], self.hi[x]])
                axes[x,y].hist2d(
                    self.data[self.variables[x]].values, 
                    self.data[self.variables[y]].values, 
                    cmap=plt.cm.YlOrRd_r,
                    bins=[40,40],
                    range=[[self.lo[x], self.hi[x]], [self.lo[y], self.hi[y]]]
                )
                
    
        ## Labels on the diagonal
        for i, label in enumerate(self.labels):
            axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center')
            axes[i,i].grid(b=False)

        # Turn on the proper x or y axes ticks.
        for i, j in zip(range(self.n), itertools.cycle((-1, 0))):
            axes[j,i].xaxis.set_visible(True)
            axes[i,j].yaxis.set_visible(True)






## Define convenient lists of columns
all_columns = [
    'Ref',
    'Ref_5x5_10th',
    'Ref_5x5_50th',
    'Ref_5x5_90th',
    'RefComposite',
    'RefComposite_5x5_10th',
    'RefComposite_5x5_50th',
    'RefComposite_5x5_90th',
    'RhoHV',
    'RhoHV_5x5_10th',
    'RhoHV_5x5_50th',
    'RhoHV_5x5_90th',
    'Zdr',
    'Zdr_5x5_10th',
    'Zdr_5x5_50th',
    'Zdr_5x5_90th',
    'Kdp',
    'Kdp_5x5_10th',
    'Kdp_5x5_50th',
    'Kdp_5x5_90th'
]

basic_columns = all_columns[:8]
polarized_columns = all_columns[8:]



## -------------------------------------------
def intervals(df):
    """
    Adds a column to the dataframe assigning a time interval for each entry
    """
    
    df['interval'] = df['minutes_past'].diff()
    
    ## Build two masks, to single out first and last rows of each group
    group_begin_mask = df['Id'].diff() != 0
    group_end_mask   = df['Id'].diff().shift(-1) != 0
    
    ## Patch the beginning entry of each group
    df['interval'][group_begin_mask] = df['minutes_past'][group_begin_mask]
    
    ## Patch the last entry of each group
    df['interval'][group_end_mask] = 60 + df['interval'][group_end_mask] - df['minutes_past'][group_end_mask]



cf = {
    'Ref' : (1.01, 1.19, 0.86),
    'RefComposite' : (1.01, 1.15, 0.87),
    'RhoHV' : (1.00, 1.05, 0.97)
}

## -------------------------------------------
def complete(df, columns=[]):
    """
    Completes the main variables using the distributions
    """
    
    for var in columns:
        df[var] = cf[var][0]*df[var].fillna(value=df['{0}_5x5_50th'.format(var)])
        df[var] = cf[var][1]*df[var].fillna(value=df['{0}_5x5_10th'.format(var)])
        df[var] = cf[var][2]*df[var].fillna(value=df['{0}_5x5_90th'.format(var)])
    
        del df['{0}_5x5_10th'.format(var)]
        del df['{0}_5x5_50th'.format(var)]
        del df['{0}_5x5_90th'.format(var)]




## ========================================
class Splitter:
    """
    A class to split the dataframe into inclusive and exclusive subsets
    that share features
    """


    ## ------------------------------------------
    def __init__(self, dataframe):
        """
        Constructor
        """

        self.df = dataframe

        ## Transform the dataset into booleans, indicating True for present values
        ## and false for missing values
        self.bool_df = self.df.notnull()
        self.bool_df['Id'] = self.df['Id']

        ## Collapse the time series in two different ways:
        ##     all: all values in time are present
        ##     any: any one value in time is present
        ##
        ## from these we can derive:
        ##     not any: no value is present

        all_df = self.bool_df.groupby('Id').agg(np.all)
        any_df = self.bool_df.groupby('Id').agg(np.any)

        self.ref_mask           = any_df['Ref']

        self.basic_all_mask     = all_df[basic_columns].all(axis=1)
        self.polarized_all_mask = all_df[polarized_columns].all(axis=1)

        self.basic_null_mask     = ~any_df[basic_columns].any(axis=1)
        self.polarized_null_mask = ~any_df[polarized_columns].any(axis=1)

        self.basic_partial_mask     = ~self.basic_null_mask & ~self.basic_all_mask
        self.polarized_partial_mask = ~self.polarized_null_mask & ~self.polarized_all_mask


    ## -------------------------------------------
    def polarized(self):
        """
        Guarantees complete information
        """

        return self.df[self.df['Id'].map(self.basic_all_mask & self.polarized_all_mask)]


    ## -------------------------------------------
    def basic(self):
        """
        Guarantees complete information for the reflectivity, but incomplete or
        absent information for the polarization values
        """

        return self.df[self.df['Id'].map(self.basic_all_mask & self.polarized_null_mask)]


    ## -------------------------------------------
    def null(self):
        """
        Gurantees that no information is present
        """

        return self.df[self.df['Id'].map(self.basic_null_mask | ~self.ref_mask)]


    ## --------------------------------------------
    def partial_basic(self):
        """
        Gurantees that only a fraction of the reflectivity data is available
        """

        return self.df[self.df['Id'].map(self.ref_mask & self.basic_partial_mask & self.polarized_null_mask)]


    ## --------------------------------------------
    def partial_polarized(self):
        """
        Guarantees that at least some basic information is present,
        and that only a partial set of polarization information is present
        """

        cross = self.basic_partial_mask | self.polarized_partial_mask
        none  = self.basic_null_mask | self.polarized_null_mask

        return self.df[self.df['Id'].map(self.ref_mask & cross & ~none)]






