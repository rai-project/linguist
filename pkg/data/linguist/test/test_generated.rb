require_relative "./helper"

class TestGenerated < Minitest::Test
  include Linguist

  class DataLoadedError < StandardError; end

  def generated_without_loading_data(blob)
    begin
      assert Generated.generated?(blob, lambda { raise DataLoadedError.new }), "#{blob} was not recognized as a generated file"
    rescue DataLoadedError
      assert false, "Data was loaded when calling generated? on #{blob}"
    end
  end

  def generated_loading_data(blob)
    assert_raises(DataLoadedError, "Data wasn't loaded when calling generated? on #{blob}") do
      Generated.generated?(blob, lambda { raise DataLoadedError.new })
    end
    assert Generated.generated?(blob, lambda { IO.read(blob) }), "#{blob} was not recognized as a generated file"
  end

  def generated_fixture_without_loading_data(name)
    generated_without_loading_data(File.join(fixtures_path, name))
  end

  def generated_fixture_loading_data(name)
    generated_loading_data(File.join(fixtures_path, name))
  end

  def generated_sample_without_loading_data(name)
    generated_without_loading_data(File.join(samples_path, name))
  end

  def generated_sample_loading_data(name)
    generated_loading_data(File.join(samples_path, name))
  end

  def test_check_generated
    # Xcode project files
    generated_sample_without_loading_data("Binary/MainMenu.nib")
    generated_sample_without_loading_data("Dummy/foo.xcworkspacedata")
    generated_sample_without_loading_data("Dummy/foo.xcuserstate")

    # Go-specific vendored paths
    generated_sample_without_loading_data("go/vendor/github.com/foo.go")
    generated_sample_without_loading_data("go/vendor/golang.org/src/foo.c")
    generated_sample_without_loading_data("go/vendor/gopkg.in/some/nested/path/foo.go")

    # .NET designer file
    generated_sample_without_loading_data("Dummu/foo.designer.cs")

    # Composer generated composer.lock file
    generated_sample_without_loading_data("JSON/composer.lock")

    # Node modules
    generated_sample_without_loading_data("Dummy/node_modules/foo.js")

    # npm shrinkwrap file
    generated_sample_without_loading_data("Dummy/npm-shrinkwrap.json")

    # Godep saved dependencies
    generated_sample_without_loading_data("Godeps/Godeps.json")
    generated_sample_without_loading_data("Godeps/_workspace/src/github.com/kr/s3/sign.go")

    # Generated by Zephir
    generated_sample_without_loading_data("C/exception.zep.c")
    generated_sample_without_loading_data("C/exception.zep.h")
    generated_sample_without_loading_data("PHP/exception.zep.php")

    # Minified files
    generated_sample_loading_data("JavaScript/jquery-1.6.1.min.js")

    # JS files with source map reference
    generated_sample_loading_data("JavaScript/namespace.js")

    # Source Map
    generated_fixture_without_loading_data("Data/bootstrap.css.map")
    generated_fixture_loading_data("Data/sourcemap.v3.map")
    generated_fixture_loading_data("Data/sourcemap.v1.map")

    # Yarn locfile
    generated_fixture_loading_data("Data/yarn.lock")

    # Specflow
    generated_fixture_without_loading_data("Features/BindingCulture.feature.cs")

    # JFlex
    generated_sample_loading_data("Java/JFlexLexer.java")

    # GrammarKit
    generated_sample_loading_data("Java/GrammarKit.java")

    # roxygen2
    generated_sample_loading_data("R/import.Rd")
  end
end