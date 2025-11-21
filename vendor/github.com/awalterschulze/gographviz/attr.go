//Copyright 2017 GoGraphviz Authors
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http)://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

package gographviz

import "fmt"

// Attr is an attribute key
type Attr string

// NewAttr creates a new attribute key by checking whether it is a valid key
func NewAttr(key string) (Attr, error) {
	a, ok := validAttrs[key]
	if !ok {
		return Attr(""), fmt.Errorf("%s is not a valid attribute", key)
	}
	return a, nil
}

const (
	// Damping http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:Damping
	Damping Attr = "Damping"
	// K http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:K
	K Attr = "K"
	// URL http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:URL
	URL Attr = "URL"
	// Background http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:_background
	Background Attr = "_background"
	// Area http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:area
	Area Attr = "area"
	// ArrowHead http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:arrowhead
	ArrowHead Attr = "arrowhead"
	// ArrowSize http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:arrowsize
	ArrowSize Attr = "arrowsize"
	// ArrowTail http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:arrowtail
	ArrowTail Attr = "arrowtail"
	// BB http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:bb
	BB Attr = "bb"
	// BgColor http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:bgcolor
	BgColor Attr = "bgcolor"
	// Center http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:center
	Center Attr = "center"
	// Charset http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:charset
	Charset Attr = "charset"
	// ClusterRank http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:clusterrank
	ClusterRank Attr = "clusterrank"
	// Color http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:color
	Color Attr = "color"
	// ColorScheme http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:colorscheme
	ColorScheme Attr = "colorscheme"
	// Comment http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:comment
	Comment Attr = "comment"
	// Compound http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:compound
	Compound Attr = "compound"
	// Concentrate http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:concentrate
	Concentrate Attr = "concentrate"
	// Constraint http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:constraint
	Constraint Attr = "constraint"
	// Decorate http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:decorate
	Decorate Attr = "decorate"
	// DefaultDist http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:defaultdist
	DefaultDist Attr = "defaultdist"
	// Dim http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:dim
	Dim Attr = "dim"
	// Dimen http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:dimen
	Dimen Attr = "dimen"
	// Dir http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:dir
	Dir Attr = "dir"
	// DirEdgeConstraints http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:dir
	DirEdgeConstraints Attr = "diredgeconstraints"
	// Distortion http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:distortion
	Distortion Attr = "distortion"
	// DPI http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:dpi
	DPI Attr = "dpi"
	// EdgeURL http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d::edgeURL
	EdgeURL Attr = "edgeURL"
	// EdgeHREF http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d::edgehref
	EdgeHREF Attr = "edgehref"
	// EdgeTarget http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d::edgetarget
	EdgeTarget Attr = "edgetarget"
	// EdgeTooltip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d::edgetooltip
	EdgeTooltip Attr = "edgetooltip"
	// Epsilon http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d::epsilon
	Epsilon Attr = "epsilon"
	// ESep http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d::epsilon
	ESep Attr = "esep"
	// FillColor http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fillcolor
	FillColor Attr = "fillcolor"
	// FixedSize http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fixedsize
	FixedSize Attr = "fixedsize"
	// FontColor http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fontcolor
	FontColor Attr = "fontcolor"
	// FontName http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fontname
	FontName Attr = "fontname"
	// FontNames http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fontnames
	FontNames Attr = "fontnames"
	// FontPath http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fontpath
	FontPath Attr = "fontpath"
	// FontSize http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:fontsize
	FontSize Attr = "fontsize"
	// ForceLabels http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:forcelabels
	ForceLabels Attr = "forcelabels"
	// GradientAngle http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:gradientangle
	GradientAngle Attr = "gradientangle"
	// Group http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:group
	Group Attr = "group"
	// HeadURL http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headURL
	HeadURL Attr = "headURL"
	// HeadLP http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:head_lp
	HeadLP Attr = "head_lp"
	// HeadClip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headclip
	HeadClip Attr = "headclip"
	// HeadHREF http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headhref
	HeadHREF Attr = "headhref"
	// HeadLabel http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headlabel
	HeadLabel Attr = "headlabel"
	// HeadPort http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headport
	HeadPort Attr = "headport"
	// HeadTarget http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headtarget
	HeadTarget Attr = "headtarget"
	// HeadTooltip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:headtooltip
	HeadTooltip Attr = "headtooltip"
	// Height http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:height
	Height Attr = "height"
	// HREF http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:href
	HREF Attr = "href"
	// ID http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:id
	ID Attr = "id"
	// Image http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:image
	Image Attr = "image"
	// ImagePath http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:imagepath
	ImagePath Attr = "imagepath"
	// ImagePos https://graphviz.org/doc/info/attrs.html#d:imagepos
	ImagePos Attr = "imagepos"
	// ImageScale http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:imagescale
	ImageScale Attr = "imagescale"
	// InputScale http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:inputscale
	InputScale Attr = "inputscale"
	// Label http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:label
	Label Attr = "label"
	// LabelURL http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelURL
	LabelURL Attr = "labelURL"
	// LabelScheme http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:label_scheme
	LabelScheme Attr = "label_scheme"
	// LabelAngle http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelangle
	LabelAngle Attr = "labelangle"
	// LabelDistance http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labeldistance
	LabelDistance Attr = "labeldistance"
	// LabelFloat http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelfloat
	LabelFloat Attr = "labelfloat"
	// LabelFontColor http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelfontcolor
	LabelFontColor Attr = "labelfontcolor"
	// LabelFontName http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelfontname
	LabelFontName Attr = "labelfontname"
	// LabelFontSize http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelfontsize
	LabelFontSize Attr = "labelfontsize"
	// LabelHREF http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelhref
	LabelHREF Attr = "labelhref"
	// LabelJust http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labeljust
	LabelJust Attr = "labeljust"
	// LabelLOC http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labelloc
	LabelLOC Attr = "labelloc"
	// LabelTarget http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labeltarget
	LabelTarget Attr = "labeltarget"
	// LabelTooltip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:labeltooltip
	LabelTooltip Attr = "labeltooltip"
	// Landscape http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:landscape
	Landscape Attr = "landscape"
	// Layer http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:layer
	Layer Attr = "layer"
	// LayerListSep http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:layerlistsep
	LayerListSep Attr = "layerlistsep"
	// Layers http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:layers
	Layers Attr = "layers"
	// LayerSelect http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:layerselect
	LayerSelect Attr = "layerselect"
	// LayerSep http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:layersep
	LayerSep Attr = "layersep"
	// Layout http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:layout
	Layout Attr = "layout"
	// Len http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:len
	Len Attr = "len"
	// Levels http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:levels
	Levels Attr = "levels"
	// LevelsGap http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:levelsgap
	LevelsGap Attr = "levelsgap"
	// LHead http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:lhead
	LHead Attr = "lhead"
	// LHeight http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:lheight
	LHeight Attr = "lheight"
	// LP http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:lp
	LP Attr = "lp"
	// LTail http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:ltail
	LTail Attr = "ltail"
	// LWidth http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:lwidth
	LWidth Attr = "lwidth"
	// Margin http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:margin
	Margin Attr = "margin"
	// MaxIter http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:maxiter
	MaxIter Attr = "maxiter"
	// MCLimit http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:mclimit
	MCLimit Attr = "mclimit"
	// MinDist http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:mindist
	MinDist Attr = "mindist"
	// MinLen http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:mindist
	MinLen Attr = "minlen"
	// Mode http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:mode
	Mode Attr = "mode"
	// Model http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:model
	Model Attr = "model"
	// Mosek http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:mosek
	Mosek Attr = "mosek"
	// NewRank http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:newrank
	NewRank Attr = "newrank"
	// NodeSep http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:nodesep
	NodeSep Attr = "nodesep"
	// NoJustify http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:nojustify
	NoJustify Attr = "nojustify"
	// Normalize http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:normalize
	Normalize Attr = "normalize"
	// NoTranslate http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:notranslate
	NoTranslate Attr = "notranslate"
	// NSLimit http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:nslimit
	NSLimit Attr = "nslimit"
	// NSLimit1 http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:nslimit1
	NSLimit1 Attr = "nslimit1"
	// Ordering http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:nslimit1
	Ordering Attr = "ordering"
	// Orientation http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:orientation
	Orientation Attr = "orientation"
	// OutputOrder http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:outputorder
	OutputOrder Attr = "outputorder"
	// Overlap http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:overlap
	Overlap Attr = "overlap"
	// OverlapScaling http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:overlap_scaling
	OverlapScaling Attr = "overlap_scaling"
	// OverlapShrink http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:overlap_shrink
	OverlapShrink Attr = "overlap_shrink"
	// Pack http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:pack
	Pack Attr = "pack"
	// PackMode http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:packmode
	PackMode Attr = "packmode"
	// Pad http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:pad
	Pad Attr = "pad"
	// Page http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:page
	Page Attr = "page"
	// PageDir http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:pagedir
	PageDir Attr = "pagedir"
	// PenColor http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:pencolor
	PenColor Attr = "pencolor"
	// PenWidth http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:penwidth
	PenWidth Attr = "penwidth"
	// Peripheries http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:peripheries
	Peripheries Attr = "peripheries"
	// Pin http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:peripheries
	Pin Attr = "pin"
	// Pos http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:pos
	Pos Attr = "pos"
	// QuadTree http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:quadtree
	QuadTree Attr = "quadtree"
	// Quantum http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:quantum
	Quantum Attr = "quantum"
	// Rank http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:rank
	Rank Attr = "rank"
	// RankDir http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:rankdir
	RankDir Attr = "rankdir"
	// RankSep http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:ranksep
	RankSep Attr = "ranksep"
	// Ratio http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:ratio
	Ratio Attr = "ratio"
	// Rects http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:rects
	Rects Attr = "rects"
	// Regular http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:regular
	Regular Attr = "regular"
	// ReMinCross http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:remincross
	ReMinCross Attr = "remincross"
	// RepulsiveForce http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:repulsiveforce
	RepulsiveForce Attr = "repulsiveforce"
	// Resolution http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:resolution
	Resolution Attr = "resolution"
	// Root http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:root
	Root Attr = "root"
	// Rotate http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:rotate
	Rotate Attr = "rotate"
	// Rotation http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:rotation
	Rotation Attr = "rotation"
	// SameHead http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:samehead
	SameHead Attr = "samehead"
	// SameTail http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:sametail
	SameTail Attr = "sametail"
	// SamplePoints http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:samplepoints
	SamplePoints Attr = "samplepoints"
	// Scale http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:scale
	Scale Attr = "scale"
	// SearchSize http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:searchsize
	SearchSize Attr = "searchsize"
	// Sep http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:sep
	Sep Attr = "sep"
	// Shape http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:shape
	Shape Attr = "shape"
	// ShapeFile http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:shapefile
	ShapeFile Attr = "shapefile"
	// ShowBoxes http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:showboxes
	ShowBoxes Attr = "showboxes"
	// Sides http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:sides
	Sides Attr = "sides"
	// Size http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:size
	Size Attr = "size"
	// Skew http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:skew
	Skew Attr = "skew"
	// Smoothing http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:smoothing
	Smoothing Attr = "smoothing"
	// SortV http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:sortv
	SortV Attr = "sortv"
	// Splines http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:splines
	Splines Attr = "splines"
	// Start http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:start
	Start Attr = "start"
	// Style http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:style
	Style Attr = "style"
	// StyleSheet http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:stylesheet
	StyleSheet Attr = "stylesheet"
	// TailURL http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tailURL
	TailURL Attr = "tailURL"
	// TailLP http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tail_lp
	TailLP Attr = "tail_lp"
	// TailClip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tailclip
	TailClip Attr = "tailclip"
	// TailHREF http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tailhref
	TailHREF Attr = "tailhref"
	// TailLabel http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:taillabel
	TailLabel Attr = "taillabel"
	// TailPort http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tailport
	TailPort Attr = "tailport"
	// TailTarget http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tailtarget
	TailTarget Attr = "tailtarget"
	// TailTooltip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tailtooltip
	TailTooltip Attr = "tailtooltip"
	// Target http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:target
	Target Attr = "target"
	// Tooltip http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tooltip
	Tooltip Attr = "tooltip"
	// TrueColor http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:tooltip
	TrueColor Attr = "truecolor"
	// Vertices http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:vertices
	Vertices Attr = "vertices"
	// ViewPort http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:viewport
	ViewPort Attr = "viewport"
	// VoroMargin http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:voro_margin
	VoroMargin Attr = "voro_margin"
	// Weight http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:weight
	Weight Attr = "weight"
	// Width http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:width
	Width Attr = "width"
	// XDotVersion http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:xdotversion
	XDotVersion Attr = "xdotversion"
	// XLabel http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:xlabel
	XLabel Attr = "xlabel"
	// XLP http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:xlp
	XLP Attr = "xlp"
	// Z http://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:z
	Z Attr = "z"

	// MinCross is not in the documentation, but found in the Ped_Lion_Share (lion_share.gv.txt) example
	MinCross Attr = "mincross"
	// SSize is not in the documentation, but found in the siblings.gv.txt example
	SSize Attr = "ssize"
	// Outline is not in the documentation, but found in the siblings.gv.txt example
	Outline Attr = "outline"
	// F is not in the documentation, but found in the transparency.gv.txt example
	F Attr = "f"
)

var validAttrs = map[string]Attr{
	string(Damping):            Damping,
	string(K):                  K,
	string(URL):                URL,
	string(Background):         Background,
	string(Area):               Area,
	string(ArrowHead):          ArrowHead,
	string(ArrowSize):          ArrowSize,
	string(ArrowTail):          ArrowTail,
	string(BB):                 BB,
	string(BgColor):            BgColor,
	string(Center):             Center,
	string(Charset):            Charset,
	string(ClusterRank):        ClusterRank,
	string(Color):              Color,
	string(ColorScheme):        ColorScheme,
	string(Comment):            Comment,
	string(Compound):           Compound,
	string(Concentrate):        Concentrate,
	string(Constraint):         Constraint,
	string(Decorate):           Decorate,
	string(DefaultDist):        DefaultDist,
	string(Dim):                Dim,
	string(Dimen):              Dimen,
	string(Dir):                Dir,
	string(DirEdgeConstraints): DirEdgeConstraints,
	string(Distortion):         Distortion,
	string(DPI):                DPI,
	string(EdgeURL):            EdgeURL,
	string(EdgeHREF):           EdgeHREF,
	string(EdgeTarget):         EdgeTarget,
	string(EdgeTooltip):        EdgeTooltip,
	string(Epsilon):            Epsilon,
	string(ESep):               ESep,
	string(FillColor):          FillColor,
	string(FixedSize):          FixedSize,
	string(FontColor):          FontColor,
	string(FontName):           FontName,
	string(FontNames):          FontNames,
	string(FontPath):           FontPath,
	string(FontSize):           FontSize,
	string(ForceLabels):        ForceLabels,
	string(GradientAngle):      GradientAngle,
	string(Group):              Group,
	string(HeadURL):            HeadURL,
	string(HeadLP):             HeadLP,
	string(HeadClip):           HeadClip,
	string(HeadHREF):           HeadHREF,
	string(HeadLabel):          HeadLabel,
	string(HeadPort):           HeadPort,
	string(HeadTarget):         HeadTarget,
	string(HeadTooltip):        HeadTooltip,
	string(Height):             Height,
	string(HREF):               HREF,
	string(ID):                 ID,
	string(Image):              Image,
	string(ImagePath):          ImagePath,
	string(ImagePos):           ImagePos,
	string(ImageScale):         ImageScale,
	string(InputScale):         InputScale,
	string(Label):              Label,
	string(LabelURL):           LabelURL,
	string(LabelScheme):        LabelScheme,
	string(LabelAngle):         LabelAngle,
	string(LabelDistance):      LabelDistance,
	string(LabelFloat):         LabelFloat,
	string(LabelFontColor):     LabelFontColor,
	string(LabelFontName):      LabelFontName,
	string(LabelFontSize):      LabelFontSize,
	string(LabelHREF):          LabelHREF,
	string(LabelJust):          LabelJust,
	string(LabelLOC):           LabelLOC,
	string(LabelTarget):        LabelTarget,
	string(LabelTooltip):       LabelTooltip,
	string(Landscape):          Landscape,
	string(Layer):              Layer,
	string(LayerListSep):       LayerListSep,
	string(Layers):             Layers,
	string(LayerSelect):        LayerSelect,
	string(LayerSep):           LayerSep,
	string(Layout):             Layout,
	string(Len):                Len,
	string(Levels):             Levels,
	string(LevelsGap):          LevelsGap,
	string(LHead):              LHead,
	string(LHeight):            LHeight,
	string(LP):                 LP,
	string(LTail):              LTail,
	string(LWidth):             LWidth,
	string(Margin):             Margin,
	string(MaxIter):            MaxIter,
	string(MCLimit):            MCLimit,
	string(MinDist):            MinDist,
	string(MinLen):             MinLen,
	string(Mode):               Mode,
	string(Model):              Model,
	string(Mosek):              Mosek,
	string(NewRank):            NewRank,
	string(NodeSep):            NodeSep,
	string(NoJustify):          NoJustify,
	string(Normalize):          Normalize,
	string(NoTranslate):        NoTranslate,
	string(NSLimit):            NSLimit,
	string(NSLimit1):           NSLimit1,
	string(Ordering):           Ordering,
	string(Orientation):        Orientation,
	string(OutputOrder):        OutputOrder,
	string(Overlap):            Overlap,
	string(OverlapScaling):     OverlapScaling,
	string(OverlapShrink):      OverlapShrink,
	string(Pack):               Pack,
	string(PackMode):           PackMode,
	string(Pad):                Pad,
	string(Page):               Page,
	string(PageDir):            PageDir,
	string(PenColor):           PenColor,
	string(PenWidth):           PenWidth,
	string(Peripheries):        Peripheries,
	string(Pin):                Pin,
	string(Pos):                Pos,
	string(QuadTree):           QuadTree,
	string(Quantum):            Quantum,
	string(Rank):               Rank,
	string(RankDir):            RankDir,
	string(RankSep):            RankSep,
	string(Ratio):              Ratio,
	string(Rects):              Rects,
	string(Regular):            Regular,
	string(ReMinCross):         ReMinCross,
	string(RepulsiveForce):     RepulsiveForce,
	string(Resolution):         Resolution,
	string(Root):               Root,
	string(Rotate):             Rotate,
	string(Rotation):           Rotation,
	string(SameHead):           SameHead,
	string(SameTail):           SameTail,
	string(SamplePoints):       SamplePoints,
	string(Scale):              Scale,
	string(SearchSize):         SearchSize,
	string(Sep):                Sep,
	string(Shape):              Shape,
	string(ShapeFile):          ShapeFile,
	string(ShowBoxes):          ShowBoxes,
	string(Sides):              Sides,
	string(Size):               Size,
	string(Skew):               Skew,
	string(Smoothing):          Smoothing,
	string(SortV):              SortV,
	string(Splines):            Splines,
	string(Start):              Start,
	string(Style):              Style,
	string(StyleSheet):         StyleSheet,
	string(TailURL):            TailURL,
	string(TailLP):             TailLP,
	string(TailClip):           TailClip,
	string(TailHREF):           TailHREF,
	string(TailLabel):          TailLabel,
	string(TailPort):           TailPort,
	string(TailTarget):         TailTarget,
	string(TailTooltip):        TailTooltip,
	string(Target):             Target,
	string(Tooltip):            Tooltip,
	string(TrueColor):          TrueColor,
	string(Vertices):           Vertices,
	string(ViewPort):           ViewPort,
	string(VoroMargin):         VoroMargin,
	string(Weight):             Weight,
	string(Width):              Width,
	string(XDotVersion):        XDotVersion,
	string(XLabel):             XLabel,
	string(XLP):                XLP,
	string(Z):                  Z,

	string(MinCross): MinCross,
	string(SSize):    SSize,
	string(Outline):  Outline,
	string(F):        F,
}
