from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
from scipy.special import loggamma
import networkx as nx

from stlearn._compat import Literal
from typing import Optional, Union
import io
#import cv2
from anndata import AnnData
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def tissue_plot(
    adata: AnnData,
    plot: str = None,
    method: str = None,

    genes: Optional[Union[str,list]] = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    cmap: str = "Spectral_r",
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    spot_size: Union[float,int] = 6.5,
    show_legend: bool = False,
    show_color_bar: bool = True,
    show_axis: bool = False,
    dpi: int = 96,
    store: bool = True,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:
    
    plt.rcParams['figure.dpi'] = dpi

    # Initialize values

    if plot == 'genes':

        if type(genes) == str:
            genes = [genes]
        colors = _gene_plot(adata,method,genes)

    elif plot == 'cluster':
        colors = _cluster_plot(adata,method)

    elif plot == 'factor_analysis':
        colors = _factor_analysis_plot(adata,method)

    

    if plot == "factor_analysis":

        n_factor = len(colors)
        plt.ioff()

        if "plots" not in adata.uns:
            adata.uns['plots'] = {}
                

        adata.uns['plots'].update({method: {}})

        for i in range(0,n_factor):
            fig, a = plt.subplots()
            vmin = min(colors[i])
            vmax = max(colors[i])
            sc = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=data_alpha,s=spot_size,marker="o",
                   vmin=vmin, vmax=vmax,cmap=plt.get_cmap(cmap),c=colors[i])

            if show_color_bar:
                cb = plt.colorbar(sc,cax = fig.add_axes([0.78, 0.3, 0.03, 0.38]))
                cb.outline.set_visible(False)
            if not show_axis:
                a.axis('off')

            # Overlay the tissue image
            a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)

            if output is not None:
                fig.savefig(output + "/factor_" + str(i+1) + ".png", dpi=dpi,bbox_inches='tight',pad_inches=0)

            if store:
                
                fig_np = get_img_from_fig(fig,dpi)

                plt.close(fig)

                current_plot = {"factor_"+str(i+1):fig_np}

                adata.uns['plots'][method].update(current_plot) 

        print("The plot stored in adata.uns['plots']['" + method + "']")



    elif (plot in ["cluster","genes"]):

        # Option for turning off showing figure
        plt.ioff()

        # Initialize matplotlib
        fig, a = plt.subplots()


        vmin = min(colors)
        vmax = max(colors)
        # Plot scatter plot based on pixel of spots
        sc = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=data_alpha,s=spot_size,marker="o",
                   vmin=vmin, vmax=vmax,cmap=plt.get_cmap(cmap),c=colors)

        if show_color_bar:
            if plot == "cluster":
                n_clus = len(colors.unique())
                cmap = plt.get_cmap(cmap)
                bounds=np.linspace(0, n_clus, n_clus+1)
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                


                cb = plt.colorbar(sc,cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
                cb.outline.set_visible(False)
            else:
            
                cb = plt.colorbar(sc,cax = fig.add_axes([0.78, 0.3, 0.03, 0.38]),cmap=cmap)
                cb.outline.set_visible(False)

                

        if not show_axis:
            a.axis('off')

        # Overlay the tissue image
        a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)

        if output is not None:
            fig.savefig(output + "/" + plot + ".png", dpi=dpi,bbox_inches='tight',pad_inches=0)

        if store:
            if plot == "genes":
                fig_np = get_img_from_fig(fig,dpi)
                plt.close(fig)
                if "plots" not in adata.uns:
                    adata.uns['plots'] = {}
                adata.uns['plots'].update({str(', '.join(genes)):fig_np})
                print("The plot stored in adata.uns['plot']['" + 
                    str(', '.join(genes)) + "']")

            elif plot == "cluster":
                fig_np = get_img_from_fig(fig,dpi)
                

                plt.close(fig)

                if "plots" not in adata.uns:
                    adata.uns['plots'] = {}
                adata.uns['plots'].update({method: fig_np})

                print("The plot stored in adata.uns['plots']['" + method + "']")


    elif plot == "trajectories":
        # Option for turning off showing figure
        
        plt.ioff()

        comp1=0
        comp2=1
        key_graph='epg'
        epg = adata.uns["pseudotimespace"]['epg']
        flat_tree = adata.uns["pseudotimespace"]['flat_tree']
        dict_nodes_pos = nx.get_node_attributes(epg,'pos')

        n_route = len(flat_tree.edges())
        

        if "plots" not in adata.uns:
            adata.uns['plots'] = {}
                

        adata.uns['plots'].update({plot: {}})

        routes = {}
        for root in flat_tree.nodes():
            routes[adata.obs[flat_tree.nodes[root]['label']+'_pseudotime'].name] = root
        adata.uns["pseudotimespace"]["routes"] = routes

        for i in range(0,n_route+1):
            pseudo="S"+str(i)+"_pseudotime"

            fig, a = plt.subplots()

            current_pseudotime = adata.obs[["node",pseudo]]
            color = []
            colors = np.array(current_pseudotime[pseudo])
            vmin = min(colors)
            vmax = max(colors)

            sc = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=0.8,s=6,marker="o",
                   vmin=vmin, vmax=vmax,cmap=plt.get_cmap("cool"),c=colors)

            if show_color_bar:
                cb = plt.colorbar(sc,cax = fig.add_axes([0.78, 0.3, 0.03, 0.38]))
                cb.outline.set_visible(False)
            if not show_axis:
                a.axis('off')
            a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)
            for edge_i in flat_tree.edges():
                branch_i_nodes = flat_tree.edges[edge_i]['nodes']

                #if branch_i_nodes[0] != edge_i[0]:
                #        branch_i_nodes = branch_i_nodes[::-1]
                
                direction_arr = []
                for node in branch_i_nodes:
                    tmp = current_pseudotime[current_pseudotime["node"]==node]
                    if len(tmp)>0:
                        direction_arr.append(tmp.iloc[:,1][0])
                

                if  len(direction_arr) > 1:
                        if not checkType(direction_arr):
                            branch_i_nodes = branch_i_nodes[::-1]


                branch_i_color = "#f4efd3"
                branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])

                edgex = branch_i_pos[:,0]
                edgey = branch_i_pos[:,1]
                a.plot(edgex,edgey,c = branch_i_color,lw=2,zorder=1)
                for j in range(0,len(edgex)):
                    a.arrow(edgex[j],edgey[j],edgex[j+1]-edgex[j],edgey[j+1]-edgey[j],color="red",length_includes_head=True,
                             head_width=10, head_length=10, linewidth=0,zorder=4)
                    if j == len(edgex)-2:
                        break

            a.scatter(adata.uns['pseudotimespace']['epg_centroids'][:,0],adata.uns['pseudotimespace']['epg_centroids'][:,1], color='pink',s=12,alpha=1,zorder=2)
                        
            for i in dict_nodes_pos.keys():
                a.text(dict_nodes_pos[i][comp1],dict_nodes_pos[i][comp2],i,color='black',fontsize = 5,zorder=3)

            # Overlay the tissue image
            a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)


            if output is not None:
                fig.savefig(output + "/" + pseudo + ".png", dpi=dpi,bbox_inches='tight',pad_inches=0)

            if store:
                
                fig_np = get_img_from_fig(fig,dpi)

                plt.close(fig)

                current_plot = {pseudo:fig_np}

                adata.uns['plots'][plot].update(current_plot) 

        print("The plot stored in adata.uns['plot']['" + plot + "']")



def _gene_plot(adata,method,genes):


    # Gene plot option

    if len(genes) == 0:
        raise ValueError('Genes shoule be provided, please input genes')
        
    elif len(genes) == 1:

        if genes[0] not in adata.var.index:
            raise ValueError(genes[0] + ' is not exist in the data, please try another gene')

        colors = adata[:, genes].X

        return colors
    else:

        for gene in genes:
            if gene not in adata.var.index:
                genes.remove(gene)
                warnings.warn("We removed " + gene + " because they not exist in the data")

            if len(genes) == 0:
                raise ValueError('All provided genes are not exist in the data')


        count_gene = adata[:,genes].to_df()


        if method is None:
            raise ValueError('Please provide method to combine genes by NaiveMean/NaiveSum/CumSum')

        if method == "NaiveMean":
            present_genes = (count_gene > 0).sum(axis=1) / len(genes)
        
            count_gene = (count_gene.mean(axis=1)) * present_genes
        elif method == "NaiveSum":
            present_genes = (count_gene > 0).sum(axis=1) / len(genes)

            count_gene = (count_gene.sum(axis=1)) * present_genes
            
        elif method == "CumSum":
            count_gene = count_gene.cumsum(axis=1).iloc[:,-1]

        colors = count_gene
        vmin = min(colors)

        return colors



def _cluster_plot(adata,method):
    
        
    colors = adata.obs[method+"_labels"]

    return colors

def _factor_analysis_plot(adata,method):

    if method == "ica":
        use_data = "X_ica"
    elif method == "ldvae":
        use_data = "X_ldvae"
    elif method == "fa":
        use_data = "X_fa"

    n_factor = adata.obsm[use_data].shape[1]
    l_colors = []
    for i in range(0,n_factor):
        colors = adata.obsm[use_data][:,i]
        vmin = min(colors)
        vmax = max(colors)
        l_colors.append(colors)

    return l_colors

def _trajectories_plot():


    return None

### Utils


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    from io import BytesIO

    fig.savefig(buf, format="png", dpi=dpi,bbox_inches='tight',pad_inches=0,transparent=True)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.asarray(Image.open(BytesIO(img_arr)))
    buf.close()
    #img = cv2.imdecode(img_arr, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img


def checkType(arr, n=2):  
  
    # If the first two and the last two elements  
    # of the array are in increasing order  
    if (arr[0] <= arr[1] and 
        arr[n - 2] <= arr[n - 1]) : 
        return True  
  
    # If the first two and the last two elements  
    # of the array are in decreasing order  
    elif (arr[0] >= arr[1] and 
          arr[n - 2] >= arr[n - 1]) : 
        return False  