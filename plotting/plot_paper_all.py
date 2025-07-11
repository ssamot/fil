from plotting import plot_paper_expriment_results_goofspiel_4
from plotting import plot_paper_expriment_results_ignore_jack_king
from plotting import plot_paper_expriment_results_ignore_jack_queen
from plotting import plot_paper_expriment_results_ignore_queen_king
from plotting import plot_paper_expriment_results_ignore_queens
from plotting import plot_paper_expriment_results_kuhn
from plotting import plot_paper_expriment_results_leduc
from plotting import plot_paper_expriment_results_liars_dice

try:
    plot_paper_expriment_results_goofspiel_4.load(True)
except:
    print("Goofspiel not plotted.")

try:
    plot_paper_expriment_results_ignore_jack_king.load(True)
except:
    print("Ignore jack king not plotted.")

try:
    plot_paper_expriment_results_ignore_jack_queen.load(True)
except:
    print("Ignore jack queen not plotted.")

try:
    plot_paper_expriment_results_ignore_queen_king.load(True)
except:
    print("Ignore queen king not plotted.")

try:
    plot_paper_expriment_results_ignore_queens.load(True)
except:
    print("Ignore queens not plotted.")

try:
    plot_paper_expriment_results_kuhn.load(True)
except:
    print("Kuhn not plotted.")

try:
    plot_paper_expriment_results_leduc.load(True)
except:
    print("Leduc not plotted.")

try:
    plot_paper_expriment_results_liars_dice.load(True)
except:
    print("Liars dice not plotted.")