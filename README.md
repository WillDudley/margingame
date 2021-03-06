`pip install margingame`

This package is mainly for just one class, `Initialise(...)`. 
You can:
<ul>
<li> Pass in non-default parameters for a custom game. </li>
<li> Get attributes, notably payoff matrices as Pandas dataframes. </li>
<li> Use the method to calculate the Nash equilibria. </li>
</ul>

See the code, notably Initialise.py, for more info.


<h1> Main Uses </h1>
<h2> Interactive payoff matrices for both players </h2>
<ol>
  <li> In a .ipynb file, run the code `from margingame.notebook.visualise import visualise`. </li>
  <li> Install any missing packages flagged by an error if there are any (this is due to a bug). </li>
  <li> Run `visualise()`. </li>
</ol>
<h3> Notes </h3>
<ul>
  <li> You can click the left margin of the output cell to expand/truncate it. </li>
  <li> Changing the domains of the payoff matrices is achieved by passing the relevant arguments into `visualise`. </li>
  <li> It's slow, I know. </li>
</ul>

<h2> Calculate the nash equlibria </h2>
<ol>
  <li> Run the code `from margingame.Initialise import Initialise'. </li>
  <li> Create your game with `Game = Initialise(...)', specifying any non-default parameters desired. </li>
  <li> Calculate the nash equlibria via support enumeration via `Game.calculate_equilibria_support_enum()`. </li>
</ol>

