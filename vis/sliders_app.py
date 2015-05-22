"""
This file demonstrates a bokeh applet, which can be viewed directly
on a bokeh-server. See the README.md file in this directory for
instructions on running.
"""

import logging

logging.basicConfig(level=logging.DEBUG)

import numpy as np

from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, Slider, TextInput, VBoxForm
from bokeh.models.glyphs import Text

from vecanalysis.sequentialembedding import SequentialEmbedding


class SlidersApp(HBox):
    """An example of a browser-based, interactive plot with slider controls."""

    extra_generated_classes = [["SlidersApp", "SlidersApp", "HBox"]]

    inputs = Instance(VBoxForm)
    text = Instance(TextInput)

    year = Instance(Slider)

    plot = Instance(Plot)
    source = Instance(ColumnDataSource)
    embeds = SequentialEmbedding(range(2005,2009))

    @classmethod
    def create(cls):
        """One-time creation of app's objects.

        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        obj = cls()

        obj.source = ColumnDataSource(dict(x=[], y=[], text=[]))

        obj.text = TextInput(
            title="Word", name='word', value='gay'
        )

        obj.year = Slider(
            title="Year", name='year',
            value=1980, start=2005, end=2008, step=1
        )

        toolset = "crosshair,pan,reset,resize,save,wheel_zoom"
        # Generate a figure container
        plot = figure(title_text_font_size="12pt",
                      plot_height=400,
                      plot_width=400,
                      tools=toolset,
                      title="Word Neighbourhood",
                      x_range=[-1, 1],
                      y_range=[-1, 1]
        )

  #      plot.line("x", "y", source=obj.source)
        glyph = Text(x="x", y="y", text="text")
        plot.add_glyph(obj.source, glyph)

        obj.plot = plot
        obj.update_data(True)

        obj.inputs = VBoxForm(
            children=[obj.text, obj.year]
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

    def input_change(self, obj, attrname, old, new):
        """Executes whenever the input form changes.

        It is responsible for updating the plot, or anything else you want.

        Args:
            obj : the object that changed
            attrname : the attr that changed
            old : old value of attr
            new : new value of attr
        """
        self.update_data(obj == self.text)

    def update_data(self, word_change):
        """Called each time that any watched property changes.

        This updates the sin wave data with the most recent values of the
        sliders. This is stored as two numpy arrays in a dict into the app's
        data source property.
        """
        if word_change:
            self.curr_reduced_embeds = self.embeds.get_reduced_word_subembeds(self.text.value)
        year = self.year.value  
        year_embed = self.curr_reduced_embeds.get_embed(year)
        self.source.data = dict(x=year_embed.m[:,0].flatten().tolist(), 
                y=year_embed.m[:,1].flatten().tolist(), 
                text=year_embed.iw)


@bokeh_app.route("/bokeh/semchange/")
@object_page("sin")
def make_sliders():
    app = SlidersApp.create()
    return app
