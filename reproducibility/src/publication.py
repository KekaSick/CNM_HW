"""Export scientific figures with a verified font floor at their PDF print size.

The plot data are never thinned or transformed. Only dense data artists may be
rasterized inside a vector PDF; axes, labels and ordinary curves remain vector.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import textwrap
import warnings

import matplotlib
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import Collection

PRINT_WIDTH_PT=170/25.4*72
PRINT_HEIGHT_PT=(297-40)/25.4*72*.78
_RAW_SAVEFIG=getattr(Figure.savefig,'_publication_original',Figure.savefig)


def _texts(fig):
    return [text for text in fig.findobj(Text) if text.get_visible() and text.get_text().strip()]


def _layout(fig,font_floor):
    axes=[ax for ax in fig.axes if ax.get_visible() and ax.get_label()!='<colorbar>']
    width=fig.get_size_inches()[0]*72
    height=fig.get_size_inches()[1]*72
    for ax in axes:
        ax.tick_params(which='both',labelsize=font_floor)
        for text in [ax.title,ax.xaxis.label,ax.yaxis.label]:
            if text.get_text() and '$' not in text.get_text():
                text.set_text(text.get_text().replace('\n',' '))
                available=(ax.get_position().width*width if text is not ax.yaxis.label
                           else ax.get_position().height*height)
                count=max(14,int(available/(font_floor*.55)))
                if len(text.get_text())>count:
                    text.set_text(textwrap.fill(text.get_text(),width=count,break_long_words=False,break_on_hyphens=False))
        if getattr(ax,'name','')=='3d':
            from matplotlib.ticker import MaxNLocator
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
                axis.set_major_locator(MaxNLocator(3))
                axis.labelpad=18
            ax.tick_params(pad=0)
            if ax.zaxis.label.get_text():
                ax.zaxis.label.set_visible(False)
                if not hasattr(ax,'_publication_zheader'):
                    ax._publication_zheader=ax.text2D(.66,.94,ax.zaxis.label.get_text(),
                        transform=ax.transAxes,ha='center',va='bottom',fontsize=font_floor)
    for text in _texts(fig):text.set_fontsize(max(font_floor,text.get_fontsize()))
    # Identical legends in a panel grid can be represented once, outside data.
    legends=[ax.get_legend() for ax in axes if ax.get_legend() is not None]
    labels=[tuple(t.get_text() for t in legend.get_texts()) for legend in legends]
    if legends and not fig.legends:
        shared=(len(legends)>1 and len(set(labels))==1)
        move=[legend for legend,names in zip(legends,labels) if shared or len(names)>=3]
        if move:
            entries={}
            for legend in move:
                for handle,text in zip(legend.legend_handles,legend.get_texts()):
                    entries.setdefault(text.get_text(),handle)
                legend.remove()
            fig.legend(list(entries.values()),list(entries),loc='lower center',
                       ncol=min(4,len(entries)),fontsize=font_floor,
                       frameon=False,bbox_to_anchor=(.5,.005))
    if not any(getattr(ax,'name','')=='3d' for ax in axes):
        # A single shared footer legend needs an explicit strip below the axes.
        fig.canvas.draw()
        bottom=max([legend.get_window_extent().y1/fig.bbox.height+.035 for legend in fig.legends],default=0.)
        rect=(0,bottom,1,1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig.tight_layout(pad=.9,w_pad=1.2,h_pad=1.3,rect=rect)
        return [str(item.message) for item in caught]
    # Tight layout is unreliable for projected 3D tick-label extents.
    fig.subplots_adjust(left=.07,right=.97,bottom=.12 if fig.legends else .07,top=.95,wspace=.4,hspace=.5)
    return []


def save_publication(fig,path,min_font_pt=8.):
    """Save PNG, vector PDF and exact physical-size QA beside the PNG."""
    import fitz
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    vector=path.with_suffix('.pdf');qa_path=path.with_suffix('.figure.json')
    matplotlib.rcParams['pdf.fonttype']=42
    matplotlib.rcParams['ps.fonttype']=42
    rasterized=0
    for artist in fig.findobj():
        dense=(isinstance(artist,Line2D) and len(artist.get_xdata())>5000)
        if isinstance(artist,Collection):
            dense=dense or len(artist.get_offsets())>5000
        if dense:artist.set_rasterized(True);rasterized+=1
    warnings_seen=[]
    scale=min(PRINT_WIDTH_PT/(fig.get_size_inches()[0]*72),
              PRINT_HEIGHT_PT/(fig.get_size_inches()[1]*72))
    floor=min_font_pt/scale*1.04
    for attempt in range(7):
        warnings_seen=_layout(fig,floor)
        fig.canvas.draw()
        for text in _texts(fig):text.set_fontsize(max(floor,text.get_fontsize()))
        extra=[label for ax in fig.axes if getattr(ax,'name','')=='3d'
               for label in [ax.xaxis.label,ax.yaxis.label]
               if label.get_visible() and label.get_text()]
        _RAW_SAVEFIG(fig,vector,bbox_inches='tight',bbox_extra_artists=extra or None,pad_inches=.06,dpi=220)
        with fitz.open(vector) as pdf:
            rect=pdf[0].rect
            size=[rect.width,rect.height]
            glyph_sizes=[span['size'] for block in pdf[0].get_text('dict')['blocks'] if block['type']==0
                         for line in block['lines'] for span in line['spans'] if span['text'].strip()]
        scale=min(PRINT_WIDTH_PT/size[0],PRINT_HEIGHT_PT/size[1])
        values=[text.get_fontsize()*scale for text in _texts(fig)]
        minimum=min(values) if values else float('inf')
        if minimum>=min_font_pt:break
        floor*=min_font_pt/minimum*1.04
    else:raise RuntimeError(f'Cannot reach {min_font_pt}pt physical font floor for {path}')
    _RAW_SAVEFIG(fig,path,bbox_inches='tight',bbox_extra_artists=extra or None,pad_inches=.06,dpi=180)
    out={'file':str(path),'vector_file':str(vector),'vector_page_pt':size,
         'target_print_box_pt':[PRINT_WIDTH_PT,PRINT_HEIGHT_PT],'print_scale':scale,
         'minimum_effective_text_pt':minimum,'required_minimum_pt':min_font_pt,
         'font_floor_definition':'Matplotlib Text base size; standard mathematical sub/superscripts are smaller',
         'minimum_emitted_glyph_span_pt':min(glyph_sizes)*scale if glyph_sizes else None,
         'text_artists':[{'text':t.get_text(),'source_pt':t.get_fontsize(),
                          'effective_pt':t.get_fontsize()*scale} for t in _texts(fig)],
         'rasterized_data_artists':rasterized,'layout_warnings':warnings_seen,
         'vector_sha256':hashlib.sha256(vector.read_bytes()).hexdigest(),
         'png_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    qa_path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out


def install_publication_exports():
    """Make existing PNG save calls also produce measured scientific vector files."""
    if getattr(Figure.savefig,'_publication_original',None) is not None:return
    def save(self,filename,*args,**kwargs):
        if isinstance(filename,(str,Path)) and Path(filename).suffix.lower()=='.png':
            return save_publication(self,filename)
        return _RAW_SAVEFIG(self,filename,*args,**kwargs)
    save._publication_original=_RAW_SAVEFIG
    Figure.savefig=save
