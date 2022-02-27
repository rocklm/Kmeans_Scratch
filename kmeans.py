'''
-----------------------------------------------------
modules
-----------------------------------------------------
'''

import numpy as np
import pandas as pd
import random


'''
-----------------------------------------------------
Kmeans class

##overview##

Kmeans is an unsupervised clustering technique
grouping data points in a dimensional space based on 
how close they are.

The number of dimensions is defined by the number of
input columns. i.e. if there are 4 columns then the algo
operates in a 4 dimensional space.

Each row in the input datset represents coordinates
in that dimensional space, ie if the datset have 4 columns
there are 4 coordinate points for each data point.

As the algo clusters based on distances, orders of magnitude,
e.g house price and square footage which are on different scales
can lead to poor clustering as the higher magnitudes will be favoured.
It is best to scale the input dataset so the magnitude of all columns/coordinates
are the same (the kmeans class provides a scaling option through standardisation).

##algo run##

The user identifies how many clusters they want and the number of iterations.

Clusters are labelled 1 to X.

Based on the number desired number of clusters X centriods are randomly created.
These are special datapoints that represent the origin/center of a cluster.
Each centriod will have the same number of coordinates as the dimensional space
which is defined by the number of input columns (ie 4 columns - centriod will have 4
coordinates).

For each data point(record) the euclidean distance is calculated with each centriod.
The the data point is then assigned to a given cluster based on it's smallest euclidean distance.

Once datapoints are assigned a cluster for an iteration - the average for each
dimension in that cluster is taken  (average of each column for that cluster) -
these averages become the coorinates for the new centriod location. 
This process repeats until the centriods do not move
(averages to not change - ie convergence) or until the max number of iterations is reached.

The cluster assignments at the final iteration or convergence are the final assignmnets

##Class Inputs##

the number of clusters.
the number of iterations.
a pandas dataframe containing only the desiered columns to cluster on.
Y/N to identify if you want to scale the input dataset or not.
#option to set random seed if required set to 9876 by default

##Class Outputs##

A dataframe with the original data, scaled data if selected and the cluster
assigments for each record.

identification of the algo converged and at what iteration as a string

the the final centriod coordinates as a dataframe

sum of squares error as a value

number of data dimensions

number of clusters

number of records

random seed

converged iteration

chosen iteration

identification if dataset was scaled

-----------------------------------------------------
'''

class Kmeans:
    
    '''
    -----------------------------------------------------
    class methods
    these are methods used within other methods
    they perform sub operations that the user does not need
    -----------------------------------------------------
    '''

    #define class inputs and variables
    def __init__(self, k, iters, df, use_scale = 'Y', seed = 9876):

        #class variables
        self.ary = np.round_(df.to_numpy(), 4)#input dataset converted to numpy array for matrix calculations
        self.rec_count = np.shape(self.ary)[0] #number of records in dataset
        self.use_scale = use_scale.upper() #flag to assess weather to scale the input dataset or not - set to Y by default
        self.final_centriods = '' #variable to store the final centriods
        self.converge = 'N' #vairable to identify if the algo converged or not, set to N by default
        self.WSS = '' #variable to store the within sum of squares of the final cluster

        #user inputs
        self.k = int(k) #number of clusters
        self.iters = int(iters) #number of iterations
        self.df = df #original dataframe
        self.dim_no = np.shape(self.ary)[1] #number of features/dimesnions/columns
        self.seed = seed #seed number to keep random generator consistent for instance

        #set seed to ensure consistent randomisation
        random.seed(self.seed)
 

    #method to scale data as different orders of magnitute
    #can bias the algo - using standardisation (mean of 0 stddev of 1)
    def scaled_data(self):

        avg = np.mean(self.ary, axis = 0)
        stddev = np.std(self.ary, axis = 0)

        return np.round_((self.ary - avg)/stddev, 4)


    #method to decide whether to use the scaled data
    #or the original input data
    #output of this method used in all other methods requiring a dataset  
    def use_scale_data(self):

        #class methods used
        #scaled_data()
        
        #if user selects Y then scale the input dataset
        if self.use_scale == 'Y':
            use_data = self.scaled_data() #class method
        else:
            use_data = self.ary

        return use_data


    #method to create random initialised centriods
    def random_centriod(self):

        #class methods used
        #use_scale_data()
        
        #initailse empty array to store coordinates for each centriod 
        centriod_array = np.array([]).reshape(self.dim_no, 0)

        rand_idx_list = []
       
        #for each centriod
        #take a random row index from the input dataset
        #and use that as the coordinates for centriod k
        #append to centriod array
        for i in range(0, self.k):
            #while loop to ensure the same random input index is not used twice
            #if the index has already been used then select another index
            #otherwise repeat the random selection
            dup_idx_break = False
            while dup_idx_break == False:
                idx = random.randint(0, self.rec_count - 1) #-1 as python index starts at 0

                if idx in rand_idx_list:
                    pass
                else:
                    dup_idx_break = True
                    break
                    
            rand_idx_list.append(idx)
            cent_i = np.array(self.use_scale_data()[idx,:])
            centriod_array = np.c_[centriod_array, cent_i]

        return np.round_(np.transpose(centriod_array), 4)
      
    
    #method to calculate the euclideana distance
    def euclid(self, centriods):

            #class methods used
            #use_scale_data()

        #identify the number of clusters
        #allows a cluster to be dropped from an iteration if not datapoints assigned
        cent_set = np.shape(centriods)[0]

        #placeholder array to store the distances between each centriod and input record
        dists = np.array([]).reshape(self.rec_count, 0)
       
        #calculate the euclidean distance between each centriod and input record
        #transposed so an array column represents a dimension (column)
        #using use_scale_data (class method) as the input dataset
        for k in range(0, cent_set):

           dist = np.transpose(np.sqrt(np.sum((self.use_scale_data() - centriods[k])**2, axis = 1)))
           dists = np.c_[dists, dist]

        return np.round_(dists, 4)


    #method to return the cluster number each record in the input dataset is assigned
    #cluster assignment based on the column index of the minimum distance
    #the cluster numbers range between 1 and K
     #as the python index start at 0 + 1 is added to get a 1 - k range
    #class method euclid() is used here to calculate the euclidean distance
    #centriods input is not a charateristic of the instance - it's a direct input which is why self is not assigned
    def clust_no(self, centriods):
       
       #class methods used
        #euclid()

        return np.argmin(self.euclid(centriods), axis = 1) + 1

     
    #method to create new centriods by averaging the coordinates of each dimension
    #for each cluster
    def avg_centriod(self, centriods):
        
        #class methods used
       # clust_no()
       #use_scaled_data()

        return np.round_(pd.DataFrame(np.c_[self.clust_no(centriods), self.use_scale_data()]).groupby(by = 0).mean().to_numpy(), 4)
    
# - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    -----------------------------------------------------
    user methods
    these are methods where the user can call
    -----------------------------------------------------
    '''

    #run Kmeans algo and return a dataset giving a cluster number to each record
    def clustered_dataset(self):

        #class methods used
        #random_centriod()
        #avg_centriod()
        #clust_no()
        #scaled_data()

        #attempt convergence within the number of iterations specified by the user
        #if the coordinates of all centriods to not change in the next iteration converganced is reached
        #and the loop is exited
        #otherwise the centriods and cluster assignments at the final itertaion is used
        for iter in range(0, self.iters):
           
           #if its the first iteration - generate random centriods
            if iter == 0:
                cur_centriods = self.random_centriod()
            else:
                pass
            #calculate the new centriod coordinates for this iteration
            new_centriods = self.avg_centriod(cur_centriods)

            #if the new centriods are the same as the previous break the loop
            #idenity the centriods have converged via setting converge = Y and idenity the iteration it occured
            #othervise the new centriods become othe current centriods for the next iteration
            if np.array_equal(new_centriods, cur_centriods) == True:
                self.converge = 'Y' #used in did_converge()
                self.converge_iter_no = iter #used in did_converge()
                
                break
            else:
                cur_centriods = new_centriods
            continue

        #pass the final centriods to the class variable
        self.final_centriods = cur_centriods
            
        #after convergence or the max number of iterations
        #create an output dataset with the original inputs, scaled inputs if selected
        #and the final cluster groups for each record in the dataset

        #identify the final cluster for each record
        clusters = pd.DataFrame(self.clust_no(cur_centriods), columns = ['Cluster'])
        
        #if the user selects scaled data
        #return a dataset with the scaled and original inputs
        if self.use_scale == 'Y':

            #for each original column name create a "name"_scld
            #and append to the scaled name list
            scaled_names = [name + '_scld' for name in self.df.columns]
            
            #create a dataframe of the scaled inputs and name the columns using the scaled names
            scaled_inputs = pd.DataFrame(self.scaled_data(), columns = scaled_names)

            #join the original dataset with the scaled names
            #otherwise just use the original input dataset
            orig_input_data = pd.concat([self.df, scaled_inputs], axis = 1)
        else:
            orig_input_data = self.df

        #join clusters with the input dataset to create the output dataset
        out_data = pd.concat([clusters, orig_input_data], axis = 1)

        # - - - - - - - - - - - - - - - - - - - - -
        #calculate the within sum of squares for final clusters
        # - - - - - - - - - - - - - - - - - - - - -

        #if user selected input dataset to be scaled
        #use the scaled data to create the sum of squares
        if self.use_scale == 'Y':
            clust_df = out_data.loc[:,out_data.columns.str.endswith('_scld')]
            clust_df = pd.concat([out_data['Cluster'], clust_df], axis = 1)
        else:
            clust_df = out_data.loc[:,~out_data.columns.str.endswith('_scld')]
         
        #placeholder variable for within sum of squares
        #each sum of squares for a given k will be added to this variable
        WSS = 0

        #for each centriod
        #get the coordinates for each cluster centriod
        #get the datapoints for each cluster
        #take the centriods away from the datapoints
        #sum and square the remainder
        #add the within sum of squars to the WSS variable
        #to get a total within cluster sum of squares
        for k in range(0, np.shape(self.final_centriods)[0]):
            clust_df_k = clust_df[clust_df['Cluster'] == k + 1].to_numpy()[:,1:]
            k_wss = np.sum((clust_df_k - self.final_centriods[k])**2, axis = None)
            
            WSS += k_wss

        #assign final WSS to class variable
        #called in WCSS method
        self.WSS = WSS

        #return clustered dataset
        return out_data


    #method to identify if the algo converged or not
    def did_converge(self):

        if self.converge == 'Y':
             print('Converged at iteration {}/{}'.\
                 format(str(self.converge_iter_no), str(self.iters)))
        else:
            print('No convergence - centriods at max iteration used')


    #method to return the final centriod coordinates as a dataframe
    def final_centriod_coords(self):

        #create a list of cluster numbers (python syntax starts at 0)
        #so index 0 is cluster 1 thus X + 1 will give us 1 for the first cluster instead of 0
        #len(self.final_centriods) counts in the number of centriods in the array
        clusts = [x + 1 for x in range(0, len(self.final_centriods))]
        #create an X1, X2 etc name for each dimension
        col_names = ['X' + str(x + 1) for x in range(0, self.dim_no)]
        #add the col name 'Clsuter' to the start of the list as Cluster will be the first column
        col_names.insert(0, 'Cluster')

        #return dataframe joining the cluster number and centriod coordinates
        return pd.DataFrame(np.c_[clusts, self.final_centriods], columns = col_names)


    #method to return within cluster sum of squares
    def WCSS(self):
        return self.WSS


    #method return number of records
    def records(self):
        return self.rec_count


    #method to return if scaling is used
    def use_scaling(self):
        return self.use_scale


    #method to return random seed for run
    def class_seed(self):
        return self.seed


    #method to return the number of dimensions of input dataset
    def data_dimensions(self):
        return self.dim_no


    #method to return the number of iterations the user selected
    def chosen_iters(self):
        return self.iters


    #method to return the number of clusters selected by the user
    def k_num(self):
        return self.k


    #method to return the seed
    def r_seed(self):
        return self.seed

'''
-----------------------------------------------------
example calls and usage
-----------------------------------------------------
'''

if __name__ == '__main__':

    #plot libraries
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import matplotlib.colors

    #import test dataset
    full_data = pd.read_csv('C:\\Side Projects\\Kmeans_scratch\\iris.csv', index_col = 0)
    data = full_data.drop(columns = 'Species')
    data = data.iloc[:,0:3]

    #call kmeans class
    test = Kmeans(3, 100, data, 'Y')
    #get the dataest with the assigned cluster for each record
    test_df = test.clustered_dataset()
    #see centriod coordinates
    test_clust = test.final_centriod_coords()
    #see if and when the algo converged
    test.did_converge()
    #see within sum of squares
    print(test.WCSS())
    #see number of record in input dataset
    print(test.records())
    #see if scaling was applied or not
    print(test.use_scaling())
    #see the random seed
    print(test.class_seed())
    #see number of clusteres selected
    print(test.k_num())
    #see number of data dimensions k means is applied on
    print(test.data_dimensions())
    #see the number of chosen iterations
    print(test.chosen_iters())
    #see random seed
    print(test.r_seed())

    #find optimal k
    k_wcss = []
    k_s = []
    for k in range(1, 10):
        k_s.append(k) 
        k_try = Kmeans(k, 100, data, 'Y')
        k_try.clustered_dataset()
        k_wcss.append(k_try.WCSS())

    plt.plot(k_wcss, linestyle = 'dotted')
    plt.show()

    #plot clusters
    ax = plt.axes(projection ="3d")
    ax.scatter3D(
        test_df['Sepal_Length_scld']
        ,test_df['Sepal_Width_scld']
        ,test_df['Petal_Length_scld']
        ,c = test_df['Cluster']
        ,cmap = matplotlib.colors.ListedColormap(['red', 'green', 'blue', 'purple'])
        )
    ax.scatter3D(
        test_clust['X1']
        ,test_clust['X2']
        ,test_clust['X3']
        ,c = test_clust['Cluster']
        ,cmap = matplotlib.colors.ListedColormap(['#4C0002', '#004E06', '#00044E', '#4C004E'])
        ,s = 200
        ,marker='^'
        )
    plt.title('3D Scatter of Kmeans')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    plt.show()

    #see how well cluster works
    #compare cluster to species category
    val = test_df.join(full_data['Species'])
    val.groupby(['Cluster', 'Species']).count()