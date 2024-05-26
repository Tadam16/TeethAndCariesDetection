import 'package:get_it/get_it.dart';

import '../../bloc/dicom_bloc.dart';
import '../../interaktor.dart';
import '../dio/dicom_api.dart';
import '../dio/dicom_api_service.dart';



final injector = GetIt.instance;

void initDependencies() {
  injector.registerSingleton<DicomApi>(DicomApiService());
  injector.registerFactory(() => DicomApiService());


  injector.registerSingletonAsync(() async {
    return Interactor(
      injector<DicomApiService>(),
    );
  });

  injector.registerFactory(
        () => DicomBloc(
      injector<Interactor>(),
    ),
  );


}
