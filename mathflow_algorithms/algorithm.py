class Algorithm():
    @staticmethod
    def get_data_requirements(method=""):
        """
        Returns a list of keys for required data to
        be fetched in the database.

        method can be used to select data keys depending on what method is going to be called
        """
        raise NotImplementedError

    def _on_call(self, data):
        """
        Any algorithm can be called, with a data dictionary containing
        values for each key returned by self.get_data_requirements("__call__").
        """
        raise NotImplementedError

    def __call__(self, data):
        """
        WARNING: This method should not be overriden
        (Override the _on_call method instead.)

        Every algorithm can be called. A data integrity check is performed,
        then the private _on_call method is executed.
        """
        return self._on_call(data)


    def update(self, data):
        pass

