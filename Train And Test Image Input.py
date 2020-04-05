X_train = []
Y_train = []
files = glob.glob (r"/Users/mjose/Desktop/Train/British_ShortHair_Train/*.JPG") # your image path
for myFile in files:
    tf.reset_default_graph()
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_train.append (image)
    X_train.append (rght_img)
    Y_train.append(["British ShortHair Cat"])
    Y_train.append(["British ShortHair Cat"])
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Train/Persian_Cat_Train/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_train.append (image)
    X_train.append (rght_img)
    Y_train.append(["Persian Cat"])
    Y_train.append(["Persian Cat"])
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Train/Maine_Coon_Train/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_train.append (image)
    X_train.append (rght_img)
    Y_train.append(["Maine Coon Cat"])    
    Y_train.append(["Maine Coon Cat"])   
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Train/Siamese_Cat_Train/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_train.append (image)
    X_train.append (rght_img)
    Y_train.append(["Siamese Cat"])    
    Y_train.append(["Siamese Cat"])
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Train/Bombay_Cat_Train/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_train.append (image)
    X_train.append (rght_img)
    Y_train.append(["Bombay Cat"])    
    Y_train.append(["Bombay Cat"])   
    
    
files = glob.glob (r"/Users/mjose/Desktop/Train/Chartreux_Cat_Train/*.JPG") # your image path

for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_train.append (image)
    X_train.append (rght_img)
    Y_train.append(["Chartreux Cat"])    
    Y_train.append(["Chartreux Cat"])       


# In[35]:


X_test = []
Y_test = []
files = glob.glob (r"/Users/mjose/Desktop/Test/British_ShortHair_Cat_Test/*.JPG") # your image path
for myFile in files:
    tf.reset_default_graph()
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_test.append (image)
    X_test.append (rght_img)
    Y_test.append(["British ShortHair Cat"])
    Y_test.append(["British ShortHair Cat"])
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Test/Persian_Cat_Test/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_test.append (image)
    X_test.append (rght_img)
    Y_test.append(["Persian Cat"])
    Y_test.append(["Persian Cat"])
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Test/Maine_Coon_Test/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_test.append (image)
    X_test.append (rght_img)
    Y_test.append(["Maine Coon Cat"])    
    Y_test.append(["Maine Coon Cat"])   
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Test/Siamese_Cat_Test/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_test.append (image)
    X_test.append (rght_img)
    Y_test.append(["Siamese Cat"])    
    Y_test.append(["Siamese Cat"])
    
    
    
files = glob.glob (r"/Users/mjose/Desktop/Test/Bombay_Cat_Test/*.JPG") # your image path
for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_test.append (image)
    X_test.append (rght_img)
    Y_test.append(["Bombay Cat"])    
    Y_test.append(["Bombay Cat"])   
    
    
files = glob.glob (r"/Users/mjose/Desktop/Test/Chartreux_Cat_Test/*.JPG") # your image path

for myFile in files:
    image = skimage.io.imread (myFile)
    image = cv2.resize(image,(227,227),interpolation=cv2.INTER_CUBIC)
    brght_img = tf.image.flip_left_right(image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    rght_img=brght_img.eval(session=sess)
    #brght_img = np.array(brght_img,dtype='int8')
    X_test.append (image)
    X_test.append (rght_img)
    Y_test.append(["Chartreux Cat"])    
    Y_test.append(["Chartreux Cat"])       




