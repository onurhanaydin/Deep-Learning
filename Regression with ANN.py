#!/usr/bin/env python
# coding: utf-8

# ## Regression with ANN

# # Keras Regression

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10,6)

sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)


# In[ ]:


# !pip install folium 


# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


#df = pd.read_csv("drive/MyDrive/Colab_Files/data/kc_house_data.csv")


# In[ ]:


#from matplotlib import style
#style.use('dark_background')


# In[ ]:


# df = pd.read_csv("../data&resources/kc_house_data.csv")


# In[2]:


#local ingest
df = pd.read_csv("/Users/onurhanaydin/Desktop/Data Science/DL/DL-S3 (Regression with ANN-kc house data)-dummy version-inclass/kc_house_data.csv")


# ## Exploratory Data Analysis and Visualization

# In[3]:


df.head()


# We will be using data from a Kaggle data set:
# 
# https://www.kaggle.com/harlfoxem/housesalesprediction
# 
# #### Feature Columns
#     
# * id - Unique ID for each home sold
# * date - Date of the home sale
# * price - Price of each home sold
# * bedrooms - Number of bedrooms
# * bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# * sqft_living - Square footage of the apartments interior living space
# * sqft_lot - Square footage of the land space
# * floors - Number of floors
# * waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# * view - An index from 0 to 4 of how good the view of the property was
# * condition - An index from 1 to 5 on the condition of the apartment,
# * grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# * sqft_above - The square footage of the interior housing space that is above ground level
# * sqft_basement - The square footage of the interior housing space that is below ground level
# * yr_built - The year the house was initially built
# * yr_renovated - The year of the house’s last renovation
# * zipcode - What zipcode area the house is in
# * lat - Lattitude
# * long - Longitude
# * sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# * sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

# In[6]:


df.info()


# In[7]:


df.isnull().sum().any()


# In[8]:


df.describe().T


# We got our paper pen notes:
#    price: like there is an outlier
#    sqft_living: outlier probability
#    sqft_above?
# 
# We will review what we have noted later.

# ### İd_number

# In[9]:


df = df.drop('id', axis = 1)


# ### price

# In[10]:


sns.distplot(df['price']);


# In[11]:


df[df["price"] > 3000000].sort_values(by="price", ascending=False)


# In[12]:


df.groupby("waterfront").mean().T


# In[13]:


plt.figure(figsize = (8,10))
df.corr()["price"].sort_values().drop("price").plot(kind = "barh");


# In[14]:


sns.scatterplot(x = 'price',y = 'sqft_living', data = df, hue = "grade");


# ### bedrooms

# In[15]:


sns.countplot(df['bedrooms']);


# Right after 33 there is 11. It is obvious that there is a problem

# In[16]:


sns.boxplot(x = 'bedrooms', y = 'price', data = df);


# States 10, 11, and 33 are weird. Prices seem low.

# In[17]:


df[df["bedrooms"] > 10]


# maybe they entered 33 by mistake instead of 3. I'm removing it right now because it's a single line.

# In[18]:


df = df[df["bedrooms"] != 33]


# Most likely the data was entered incorrectly.

# In[19]:


df.shape


# ### date

# date cannot be used. I will try to divide it into months and years and use it.

# In[20]:


df['date'].dtype


# In[21]:


df['date'] = pd.to_datetime(df['date'])


# In[22]:


df['date']


# In[23]:


df['year'] = df['date'].dt.year
#df['year'] = df['date'].apply(lambda date : date.year)


# In[24]:


df['month'] = df['date'].dt.month


# I have parsed the date data. Became more usable.

# In[25]:


df.head()


# In[26]:


sns.boxplot(x = 'year', y = 'price', data = df);


# In[27]:


df.groupby('year')['price'].mean().plot();


# There is a difference of 25,000. they don't have many different values.
# 
# It does not make sense to build our model on the year. new data is coming every year, our model will not use the previous years.

# In[28]:


sns.boxplot(x = 'month', y = 'price', data = df);


# In[29]:


df.groupby('month')['price'].mean().plot();


# Prices may change according to the season. Let's review later.
# 
# How can I use the month information? dummy is the most logical. I can't use it as a number.

# In[30]:


month_dummy = pd.get_dummies(df["month"], prefix = "month")
df = pd.concat([df, month_dummy], axis = 1)
df.head()


# In[31]:


df = df.drop(['date', "year", "month"], axis = 1)


# ### zipcode

# In[32]:


df['zipcode'].value_counts(dropna = False)


# I have to do zip code get_dummies. but the number of variables is increasing a lot. not available. If we could divide into regions, we could group. I'm handing.

# we can categorize the zipcodes as north, south, west, east, middle by regions. But it can be made manually and taken many time and we need domain knowladge to do that. So we will drop this column.

# In[33]:


df = df.drop('zipcode', axis = 1)


# ### yr_renovated & yr_built

# In[34]:


df['yr_renovated'].value_counts(dropna = False)


# 0 : unrenovated houses.

# In[35]:


df['yr_built'].value_counts(dropna = False)


# It may force my model as the year values converge with scale.

# could make sense due to scaling, higher should correlate to more value

# If the house has not been renovated, we apply the build date.

# In[ ]:


# df["yr_renovated"].replace(0, np.nan, inplace = True)
# df["yr_renovated"].fillna(df["yr_built"], inplace = True)
# df.drop("yr_built", axis = 1, inplace = True)


# In[ ]:


# df["yr_renovated"].replace(0, np.nan, inplace = True)
# df["yr_renovated"].fillna(df["yr_built"], inplace = True)
# df["new_age"] = 2021 - df["yr_renovated"]
# df.drop(["yr_renovated", "yr_built"], axis = 1, inplace = True)


# ### sqft_basement

# In[36]:


sns.distplot(df['sqft_basement']);


# In[37]:


df['sqft_basement'].value_counts(dropna = False)


# 0: no basement

# In[38]:


df[df["sqft_basement"] > 3000].sort_values(by="sqft_basement", ascending=False)


# could make sense due to scaling, higher should correlate to more value

# ### sqft_above

# In[39]:


sns.distplot(df['sqft_above']);


# In[40]:


df['sqft_above'].value_counts(dropna = False)


# In[41]:


df[df["sqft_above"] > 6000].sort_values(by="sqft_above", ascending=False)


# no probe is visible in the totals.
# I'm looking at the ones that are disproportionate to the number of rooms.

# ### Geographical Properties

# In[42]:


plt.figure(figsize = (12, 8))
sns.scatterplot(x = 'price', y = 'long', data = df);


# I'm keeping up. -122 longitude has high house prices.

# In[43]:


plt.figure(figsize  = (12, 8))
sns.scatterplot(x = 'price', y = 'lat', data = df);


# I see that there are high house prices in latitude as well.
# 
# Let me look at both.

# In[44]:


plt.figure(figsize = (12, 8))
sns.scatterplot(x = 'long', y = 'lat', data = df, hue = 'price');


# dark colors are high priced ones.
# 
# to see more clearly,
# 
# Let's see what 1% is.

# In[45]:


len(df) * (0.01)


# In[49]:


df.sort_values('price', ascending = False).head(216)


# Let me create a new variable..
# I'll take all but the first 216:

# In[50]:


non_top_1_perc = df.sort_values('price', ascending = False).iloc[216:]


# Let me visualize accordingly..

# In[51]:


plt.figure(figsize = (12, 8))
sns.scatterplot(x = 'long', y = 'lat', data = non_top_1_perc, hue = 'price',
                palette = 'RdYlGn', edgecolor = None, alpha = 0.2);


# green ones are more expensive. he became able to see the new expensive houses and the differences.
# 
# there are cheap houses by the sea (varosh).

# In[46]:


sns.boxplot(x = 'waterfront', y = 'price', data = df);


# seaside houses seem more expensive.

# In[4]:


import folium
folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=10)


# All places in the world are registered in the folium. My data is not visible on it. Let's place them.

# In[5]:


map_kc = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start = 10) # location=[47.511,  -122.257]
for lat, lng in zip(df['lat'], df['long']):
    folium.CircleMarker(
        [lat, lng],
        radius = 1,
        color = 'blue',
        fill = False ,
        fill_color ='#3186CC',
        fill_opacity = 0.3).add_to(map_kc)
map_kc


# In[6]:


map_kc = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start = 10) # location=[47.511,  -122.257]
for lat, lng, prc in zip(df['lat'], df['long'], df['price']):
     if prc > df['price'].mean():
       color = 'red'
     else : 
       color = 'blue'
     folium.CircleMarker(
        [lat, lng],
        radius = 1,
        color = color,
        fill = False ,
        fill_color = '#3186CC',
        fill_opacity = 0.3).add_to(map_kc)
map_kc


# In[136]:


df['price'].min()


# In[138]:


df['price'].max()


# In[140]:


df.describe().T


# To paint each dot a different color:
# 
# https://medium.com/datasciencearth/map-visualization-with-folium-d1403771717

# In[7]:


import branca.colormap as cm

min_price = df['price'].min()
max_price = df['price'].max()

map_kc = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start = 10) # location=[47.511,  -122.257]

myColors = cm.StepColormap(colors=['green','yellow','orange','red'],  
                           #index << min, 25%, 50%, 75%, max
                           index=[min_price,322000,450000,645000,max_price], 
                           vmin= min_price,
                           vmax=max_price)

for loc, prc in zip(zip(df['lat'], df['long']), df['price']):
      folium.CircleMarker(
        loc,
        radius = 1,
        fill=True, 
        color= myColors(prc)
      ).add_to(map_kc)
map_kc


# ### latest data

# In[61]:


df.head()


# In[62]:


df.shape


# ## Preprocessing of Data
# - Train | Test Split, Scalling

# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X = df.drop('price', axis = 1)
y = df['price']


# In[65]:


seed = 101


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)


# In[67]:


from sklearn.preprocessing import MinMaxScaler  # RobustScaler()

# If there are too many outliers in the data, robust scaler should be used, otherwise minmax can be used.


# In[68]:


scaler = MinMaxScaler()


# In[69]:


X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Modelling & Model Performance

# In[70]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# In[71]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


# In[72]:


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    score = r2_score(actual, pred)
    return print("r2_score:", score, "\nmae:", mae, "\nmse:", mse, "\nrmse:", rmse)


# In[73]:


X_train.shape


# In[74]:


tf.random.set_seed(seed)

model = Sequential()

model.add(Dense(29, activation = 'relu', input_dim = X_train.shape[1]))
#model.add(Activation("relu")) # Activation function can be added separately as a different line after each layer.  
model.add(Dense(29, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')


# we wouldn't actually be able to look up model.weights if we didn't type input_dim. in this case, we are looking at it.. otherwise we have to wait for it to be fit.

# In[76]:


#in case of the model has input_dim variable call:
model.weights


# I do my split process in fit.
# It will distribute the data for each epoch, process the pieces one by one...
# 
# batch=128 >> Calculates the rate after 128.
# 
# for each epoch:
# * pattern created with first 128 lines (there are errors).
# * gets a score from the resulting model. Checking overfit status.
# * writes the score from the last transaction to our list.
# 
# 128 x 130

# In[77]:


model.fit(x = X_train, y = y_train, validation_split = 0.15, batch_size = 128, epochs = 1000)


# In[78]:


model.summary()


# In[79]:


model.weights


# In[80]:


pd.DataFrame(model.history.history)


# In[81]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[82]:


model.evaluate(X_test, y_test, verbose=0)


# In[83]:


y_pred = model.predict(X_test)


# In[84]:


eval_metric(y_test, y_pred)


# ### learning_rate

# In[86]:


from tensorflow.keras.optimizers import Adam


# note: set_seed and model.add will be run in the same place. otherwise it takes different values. We try different seeds and try to get good values.
# 
# lr = increase tenfold, look again. if he can't learn, start giving down values... i slowly lowered it to 0.003.

# In[87]:


tf.random.set_seed(seed)

model = Sequential()

model.add(Dense(29, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(29, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

opt = Adam(lr = 0.003) # default learning rate value is 0.001
model.compile(optimizer = opt, loss = 'mse')


# In[88]:


model.weights


# In[89]:


model.fit(x = X_train, y = y_train, validation_split = 0.15, batch_size = 128, epochs = 1000)


# In[90]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[91]:


y_pred = model.predict(X_test)


# In[92]:


eval_metric(y_test, y_pred)


# ### EarlyStopping

# In[93]:


from tensorflow.keras.callbacks import EarlyStopping


# In[94]:


tf.random.set_seed(seed)

model = Sequential()

model.add(Dense(29, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(29, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

opt = Adam(lr = 0.003)
model.compile(optimizer = opt, loss = 'mse')


# look/observe val_los while the model is being trained.
# 
# patience: I'm asking you to be patient for how many values. If it can't find a better value than mine for 25 rows, it will stop. It is usually written between 20-25 on average. depends on the size of the data. If it deviates, it should be given 5. it means she can't be patient..

# In[95]:


early_stop = EarlyStopping(monitor = "val_loss", mode = "auto", verbose = 1, patience = 25)


# In[96]:


model.fit(x = X_train, y = y_train, validation_split = 0.15, batch_size = 128, epochs = 1000, callbacks = [early_stop])


# In[97]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[98]:


y_pred = model.predict(X_test)


# In[99]:


eval_metric(y_test, y_pred)


# My values have fallen. early_stop didn't work. couldn't learn better. Because I said 25, he couldn't get out of the way he was stuck. maybe it would be out after 50, it didn't.
# 
# You don't need early_stop for this data anyway.
# Overfit was not visible before. We will repeat later.

# ### Dropout

# I use it to prevent overfit.

# The Dropout layer randomly sets input units to 0 with a frequency of `rate`
# at each step during training time, which helps prevent overfitting.

# In[100]:


from tensorflow.keras.layers import Dropout


# I added a new line.
# 
# dropout: run more than 20% per iteration.
# It prevents overfit by closing another part each time.
# 
# We don't have to put it on every layer. we do it by trying. Overfit is not visible in this data.

# In[101]:


tf.random.set_seed(seed)

model = Sequential()

model.add(Dense(29, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(29, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = Adam(lr = 0.003)
model.compile(optimizer = opt, loss = 'mse')


# In[102]:


early_stop = EarlyStopping(monitor = "val_loss", mode = "auto", verbose = 1, patience = 25)


# In[103]:


model.fit(x = X_train, y = y_train, validation_split = 0.15, batch_size = 128, epochs = 1000, callbacks = [early_stop])


# In[104]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# Score dropped because I disrupted the learning with dropout.

# In[105]:


y_pred = model.predict(X_test)


# In[106]:


eval_metric(y_test, y_pred)


# ## Saving Final Model and Scaler

# In[107]:


import pickle
pickle.dump(scaler, open("scaler_kc_house", 'wb'))


# In[108]:


tf.random.set_seed(seed)

model = Sequential()

model.add(Dense(29, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(29, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

opt = Adam(lr = 0.003)
model.compile(optimizer = opt, loss = 'mse')


# I no longer need early_stop. I am running my model according to the values I have obtained above. The only change compared to the previous one is the number of lines.

# In[ ]:


#early_stop = EarlyStopping(monitor = "val_loss", mode = "auto", verbose = 1, patience = 25)


# In[109]:


model.fit(x = X_train, y = y_train, validation_data = (X_test, y_test), batch_size = 128, epochs = 1000,
         # callbacks = [early_stop]
         )


# In[118]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[119]:


y_pred = model.predict(X_test)


# In[120]:


eval_metric(y_test, y_pred)


# In[121]:


model.save('model_kc_house.h5')  # creates a HDF5 file 'my_model.h5'


# ## Loading Model and Scaler

# In[122]:


from tensorflow.keras.models import load_model


# In[123]:


model_kc_house = load_model('model_kc_house.h5')
scaler_kc_house = pickle.load(open("scaler_kc_house", "rb"))


# ## Prediction

# Let's look at the house on my first line, I used the current one for convenience.

# In[124]:


single_house = df.drop('price', axis = 1).iloc[0:1, :]
single_house


# In[125]:


single_house = scaler_kc_house.transform(single_house)
single_house


# In[126]:


model_kc_house.predict(single_house)


# In[127]:


df.iloc[0][0]


# ## Comparison with ML

# ### Linear Regression

# In[128]:


from sklearn.linear_model import LinearRegression 


# In[129]:


ln_model = LinearRegression()
ln_model.fit(X_train, y_train)
y_pred = ln_model.predict(X_test)
eval_metric(y_test, y_pred)


# ### Random Forest

# In[130]:


from sklearn.ensemble import RandomForestRegressor


# In[131]:


rf_model = RandomForestRegressor(random_state = seed)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
eval_metric(y_test, y_pred)

