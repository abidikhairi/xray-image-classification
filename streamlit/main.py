from multipage import MultiPage
from pages import prediction
from pages import analysis


app = MultiPage()

app.add_page('New Predicition', prediction.app)
app.add_page('Dataset Analysis', analysis.app)


app.run()

