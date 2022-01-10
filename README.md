# Heroku_Services
A lil repo for the mini-projects that I learnt, built, and played around with.

---

### NASA Spacetagram
A small micro-service that uses NASA's APOD API to fetch the "Atronomy Picture of the Day" and displays the result in a simplified Instagram-like interface. The interface is built with the form from the Streamlit API.

The services is hosted on Heroku at <a href="https://nasa-spacetagram.herokuapp.com/">**NASA Spacetagram**</a>. 

Please note that Heroku may put the service to sleep after a period of inactivity, so you may experience some sloooooow loading on your initial visit (assuming that none others have accessed the site for a while).

---

### Facebook DETR
This is a small micro-service that runs an end-to-end object detection task using Facebook's pre-trained DETR model. You may find details about DETR at Facebook's repo <a href="https://github.com/facebookresearch/detr">here</a>.

The service is hosted on Heroku at <a href="https://fierce-everglades-62387.herokuapp.com/">**Object Detection**</a>.

The interface is simple: the user inputs a link to an image and the model will detect objects based on the classes it was trained with.

The sad truth (meh :/) about this micro-service is that, since it's running on a free Heroku account, it's not given enough memory to handle the model for most of the time. But here's a screenshot of it working :)

<img src="https://raw.githubusercontent.com/kxchiu/Heroku_Services/main/Facebook_DETR/demo.jpg" width=600 alt="Facebook DETR Demo"/>
