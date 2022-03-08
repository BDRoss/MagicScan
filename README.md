# MagicScan

## Project Goals
I'm putting this project together to practice and learn, and hopefully to put together a solution to organizing my Magic(TM) collection in a way that existing solutions have not satisfied.

Currently this project is in it's infancy - I am still learning some new (to me) technologies that will make this possible.  At present, a lot of the code has been lifted from other developers, because I wanted to see actual working code, and have access to change things around and really figure out how it worked.  

     ####################################  
     # Current to-do list for project,  #  
     # updated as-needed.		   	    #  
     ####################################  
 



# Current TODOs to establish project  
-Look into skipping OpenCV homography, just find square  
-If square is found, may not be necessary to warp perspective - just isolate titles and set or set and set#  
	-Found out how to isolate parts of image, but haven't squared up arbitrary scans - yet  
-Tesseract font training?  
	-Found training data somebody else put together - it works better than tesseract's base  
		english data, but is still not perfect.  Probably need to train data myself.  
-Seriously, how to do font training?  
	-Gatherer images for sure  
	-Figured out how to isolate titles, can pull down a bunch of card images and try to label with Gatherer api  
-Further organize readme with a tasklist  


# Future Requirements  
-Once I can successfully isolate and read title/set# text, verify via the gatherer + pull metadata  
-Run scan on live video to allow fast scanning  
-Build android app - UI + get everything running on app, as opposed to independent python scripts  
-  