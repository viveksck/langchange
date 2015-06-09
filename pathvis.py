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

embeds = SequentialEmbedding(range(1900,2001, 5))
curr_word_path = None
curr_basis = None
curr_word = None
curr_basis_year = None


def get_data(word, basis):
    global curr_word_path, curr_basis, curr_word, curr_basis_year
    if not (word is None) or not (basis is None):
        if basis is None:
            basis = curr_basis_year
        else:
            curr_basis_year = basis
        if word is None:
            word = curr_word
        else:
            curr_word = word
        curr_word_path, curr_basis = embeds.get_word_path(curr_word, n=3, basis_year=curr_basis_year, word_list=None)
    return curr_word_path, curr_basis


class SlidersApp(HBox):
    """An example of a browser-based, interactive plot with slider controls."""

    extra_generated_classes = [["SlidersApp", "SlidersApp", "HBox"]]

    inputs = Instance(VBoxForm)
    text = Instance(TextInput)
    year = Instance(Slider)
    plot = Instance(Plot)
    word_source = Instance(ColumnDataSource)
    target_source = Instance(ColumnDataSource)
    line_source = Instance(ColumnDataSource)
    basis = Instance(Slider)

    @classmethod
    def create(cls):
        """One-time creation of app's objects.

        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        obj = cls()

        obj.word_source = ColumnDataSource(dict(x=[], y=[], text=[]))
        obj.target_source = ColumnDataSource(dict(tx=[], ty=[], ttext=[]))
        obj.line_source = ColumnDataSource(dict(word_path_x=[], word_path_y=[]))

        obj.text = TextInput(
            title="Word", name='word', value='gay'
        )

        obj.year = Slider(
            title="Year", name='year',
            value=1900, start=1900, end=2000, step=5
        )

        obj.basis = Slider(
            title="Basis Year", name='basis',
            value=2000, start=1900, end=2000, step=5
        )


        toolset = "crosshair,pan,reset,resize,save,wheel_zoom"
        # Generate a figure container
        plot = figure(title_text_font_size="12pt",
                      plot_height=400,
                      plot_width=400,
                      tools=toolset,
                      title="Word Path",
                      x_range=[-1, 1],
                      y_range=[-1, 1]
        )

  #      plot.line("x", "y", source=obj.source)
        glyph = Text(x="x", y="y", text="text", text_color="#96deb3")
        plot.add_glyph(obj.word_source, glyph)
        glypht = Text(x="tx", y="ty", text="ttext")
        plot.add_glyph(obj.target_source, glypht)
        glyphline = Line(x="word_path_x", y="word_path_y", line_width=4, line_alpha=0.3, line_color='red')
        plot.add_glyph(obj.line_source, glyphline)

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
        curr_path, curr_embed = get_data(word, basis)
        tind = curr_embed.wi[self.text.value]
        all_but = np.delete(curr_embed.m, tind, 0)
        all_but_words = [neighbour for neighbour in curr_embed.iw if not neighbour == self.text.value]
        self.word_source.data = dict(x=all_but[:,0].flatten().tolist(), 
                y=all_but[:,1].flatten().tolist(), 
                text=all_but_words)
        self.target_source.data = dict(tx=[point[0] for iyear, point in curr_path.items() if iyear==year], 
                ty=[point[1] for iyear, point in curr_path.items() if iyear==year],
                ttext = [self.text.value])
        self.line_source.data = dict(word_path_x=[point[0] for iyear, point in curr_path.items() if iyear <= year],
        word_path_y=[point[1] for iyear, point in curr_path.items() if iyear <= year])



@bokeh_app.route("/bokeh/pathchange/")
@object_page("sin")
def make_sliders():
    app = SlidersApp.create()
    return app
