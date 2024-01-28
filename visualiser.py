import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from distributions import Distribution



class DistributionVisualizer:

    def __init__(self, distribution: Distribution):
        
        self.distribution = distribution


    def plot(self, samples):
        if samples.shape[1] == 1:
            self._plot_1d(samples)
        elif samples.shape[1] > 1:
            self._plot_2d(samples)
        else:
            raise ValueError("Samples dimension not supported")


    def _plot_1d(self, samples):

        plt.hist(samples, density=True, bins=30, alpha=0.5)

        x = np.linspace(min(samples), max(samples), 100)
        y = self.distribution.density(x)
        plt.plot(x, y, 'r')

        plt.show()


    def _plot_2d(self, samples):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        hist, xedges, yedges = np.histogram2d(samples[:,0], samples[:,1], bins=30, density=True)
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        plt.show()




class HMCVisualizer(DistributionVisualizer):

    def __init__(self, distribution, sampler):
        super().__init__(distribution)
        self.sampler = sampler

    def plot_trajectory(self, initial_point, num_steps):
        points, momentums = self.sampler.sample(initial_point, num_steps)

        if points.ndim == 1:
            self._plot_1d_trajectory(points, momentums)
        elif points.ndim == 2:
            self._plot_2d_trajectory(points, momentums)
        else:
            raise ValueError("Points dimension not supported")

    def _plot_1d_trajectory(self, points, momentums):
        plt.plot(points, 'b-')
        plt.quiver(np.arange(len(points)), points, np.zeros_like(points), momentums, angles='xy', scale_units='xy', scale=1)
        plt.show()

    def _plot_2d_trajectory(self, points, momentums):
        X, Y = np.meshgrid(np.linspace(min(points[:,0]), max(points[:,0]), 100), np.linspace(min(points[:,1]), max(points[:,1]), 100))
        Z = self.distribution.density(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)

        plt.figure()
        plt.contour(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 100), cmap=cm.coolwarm)
        plt.quiver(points[:,0], points[:,1], momentums[:,0], momentums[:,1])
        plt.plot(points[:,0], points[:,1], 'ro-')

        plt.show()


if __name__ == "__main__":

    import scipy.stats as st
    

    """
    Visualise
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    dist_gen_spec = Multi_Modal_Dist_Generator(distributions, dist_parameter_dicts, size)


    dist_gen_spec.create_data()
    spec_samples = (dist_gen_spec.X, dist_gen_spec.y)
    #spec_samples = dist_gen_spec.prepare_data(0.2)

    visualiser = RawVisualiser()

    visualiser.plot_2d_scatter(spec_samples, 0, n-1)
    visualiser.plot_3d_scatter(spec_samples, 0, 1, 2)
    """


    """
    Plotly Experiments
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    df = px.data.iris()
    #print(df)
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", symbol="species")
    
    fig.show()

    gapminder = px.data.gapminder()
    gapminder2=gapminder.copy(deep=True)
    gapminder['planet']='earth'
    gapminder2['planet']='mars'
    gapminder3=pd.concat([gapminder, gapminder2])

    fig = px.bar(gapminder3, x="continent", y="pop", color="planet",
    animation_frame="year", animation_group="country", range_y=[0,4000000000*2])
    #fig.show()
    """