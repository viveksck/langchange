"""
This file demonstrates a bokeh applet, which can be viewed directly
on a bokeh-server. See the README.md file in this directory for
instructions on running.
"""

import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)


from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, Slider, TextInput, VBoxForm
from bokeh.models.glyphs import Text, Line

from vecanalysis.sequentialembedding import SequentialEmbedding

years = range(1900, 2001, 10)
embeds = SequentialEmbedding(years)
curr_reduced_embeds = None
curr_basis = None
curr_word = None

word_list = ["cheerful", "gaiety", "jolly", "genial",  "pleasant", "lesbian",  "homosexual", "gay", "mannish", "cohabiting", "rights"]

def get_data(year, word, basis):
    global curr_reduced_embeds, curr_basis, curr_word
    if not (word is None) or not (basis is None):
        if basis is None:
            basis = curr_basis
        else:
            curr_basis = basis
        if word is None:
            word = curr_word
        else:
            curr_word = word
        curr_reduced_embeds = embeds.get_reduced_word_subembeds(curr_word, n=5, basis=curr_basis, word_list=word_list)
    return curr_reduced_embeds


class SlidersApp(HBox):
    """An example of a browser-based, interactive plot with slider controls."""

    extra_generated_classes = [["SlidersApp", "SlidersApp", "HBox"]]

    inputs = Instance(VBoxForm)
    text = Instance(TextInput)
    year = Instance(Slider)
    basis = Instance(Slider)
    plot = Instance(Plot)
    source = Instance(ColumnDataSource)
    path_source = Instance(ColumnDataSource)

    @classmethod
    def create(cls):
        """One-time creation of app's objects.

        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        obj = cls()

        obj.source = ColumnDataSource(dict(x=[], y=[], text=[]))
        obj.path_source = ColumnDataSource(dict(xs=[], ys=[]))

        obj.text = TextInput(
            title="Word", name='word', value='gay'
        )

        obj.year = Slider(
            title="Display Year", name='year',
            value=1900, start=1900, end=2000, step=10
        )

        obj.basis = Slider(
            title="Basis Year", name='basis',
            value=2000, start=1900, end=2000, step=20
        )

        toolset = "crosshair,pan,reset,resize,save,wheel_zoom"
        # Generate a figure container
        plot = figure(title_text_font_size="12pt",
                      plot_height=400,
                      plot_width=400,
                      tools=toolset,
                      title="Changing Word Neighbourhood",
                      x_range=[-1, 1],
                      y_range=[-1, 1]
        )

  #      plot.line("x", "y", source=obj.source)
        glyph = Text(x="x", y="y", text="text", text_color="#96deb3", text_align='center')
        plot.add_glyph(obj.source, glyph)
        glypht = Text(x="tx", y="ty", text="ttext")
        plot.add_glyph(obj.source, glypht)

        rs = np.random.randint(150, size=len(word_list)).tolist()
        gs = np.random.randint(150, size=len(word_list)).tolist()
        bs = np.random.randint(150, size=len(word_list)).tolist()
        colors = ["#%02x%02x%02x" % (r, g, b) for r, g, b in zip(rs, gs, bs)]
        for word_i in range(len(word_list)):
            word = word_list[word_i]
            glyphline = Line(x=word+"x", y=word+"y", line_width=4, line_alpha=0.2, line_color=colors[word_i])
            plot.add_glyph(obj.path_source, glyphline)

        obj.plot = plot
        obj.update_data(True, True)

        obj.inputs = VBoxForm(
            children=[obj.text, obj.year, obj.basis]
        )

        obj.children.append(obj.inputs)
        obj.children.append(obj.plot)
    
        return obj

    def setup_events(self):
        """Attaches the on_change event to the value property of the widget.

        The callback is set to the input_change method of this app.
        """
        super(SlidersApp, self).setup_events()
        if not self.text:
            return

        # Text box event registration
        self.text.on_change('value', self, 'input_change')

        # Slider event registration:
        getattr(self, 'year').on_change('value', self, 'input_change')

        getattr(self, 'basis').on_change('value', self, 'input_change')

    def input_change(self, obj, attrname, old, new):
        """Executes whenever the input form changes.

        It is responsible for updating the plot, or anything else you want.

        Args:
            obj : the object that changed
            attrname : the attr that changed
            old : old value of attr
            new : new value of attr
        """
        self.update_data(obj == self.text, obj == self.basis)

    def update_data(self, word_change, basis_change):
        """Called each time that any watched property changes.

        This updates the sin wave data with the most recent values of the
        sliders. This is stored as two numpy arrays in a dict into the app's
        data source property.
        """
        word = None
        basis = None
        if word_change:
            word = self.text.value
        if basis_change: 
            basis = self.basis.value
        year = self.year.value  
        year_embeds = get_data(year, word, basis)
        year_embed = year_embeds.get_embed(year)
        year_embed.m -= year_embed.represent(self.text.value)
        t = year_embed.represent(self.text.value)
        tind = year_embed.wi[self.text.value]
        all_but = np.delete(year_embed.m, tind, 0)
        all_but_words = [neighbour for neighbour in year_embed.iw if not neighbour == self.text.value]
        self.source.data = dict(x=all_but[:,0].flatten().tolist(), 
                y=all_but[:,1].flatten().tolist(), 
                text=all_but_words,
                tx=[t[0]], ty=[t[1]], ttext=[self.text.value])
        word_paths_xs = []
        word_paths_ys = []
        word_paths = year_embeds.get_word_paths(word_list)
        for word in word_list:
            word_paths_xs.append([point[0] - t[0] for iyear, point in word_paths[word].items() if iyear <= year]) 
            word_paths_ys.append([point[1] - t[1] for iyear, point in word_paths[word].items() if iyear <= year]) 

        self.path_source.data = {}
        for i in range(len(word_list)):
            self.path_source.data[word_list[i]+"x"]=word_paths_xs[i]
            self.path_source.data[word_list[i]+"y"]=word_paths_ys[i]


@bokeh_app.route("/bokeh/semchange/")
@object_page("sin")
def make_sliders():
    app = SlidersApp.create()
    return app
