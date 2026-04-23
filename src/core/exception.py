import sys


class AgenticRagException(Exception):
    """Exception personnalisée pour les erreurs liées à la RAG Agentique.
        
        Attributes
         ----------
        error_message : str
            Description textuelle de l'erreur.
        filename : str
            Nom du script Python où l'exception a été levée.
        lineno : int
            Numéro de la ligne ayant causé l'erreur.
        
        """

    def __init__(self, 
                 error_message: str, 
                 error_details: sys) -> None:
        """Initialise l'exception avec les détails de la trace système.

        Parameters
        ----------
        error_message : str
            Le message d'erreur explicatif.
        error_details : sys
            Le module sys de Python, utilisé pour extraire les informations 
            de l'exception via `exc_info()`.
        """
        
        self.error_message = error_message
        _, _, exc_tab = error_details.exc_info()

        self.lineno = exc_tab.tb_lineno
        self.filename = exc_tab.tb_frame.f_code.co_filename

    def __str__(self) -> str:
        """Formate le message d'erreur complet pour l'affichage.

        Returns
        -------
        str
            Une chaîne de caractères détaillée incluant le fichier, 
            la ligne et le message d'erreur.
        """
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.filename, self.lineno, str(self.error_message)
        )